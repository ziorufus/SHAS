import argparse
from dataclasses import dataclass
from multiprocessing import cpu_count
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from constants import HIDDEN_SIZE, TARGET_SAMPLE_RATE
from data import FixedSegmentationDatasetNoTarget, segm_collate_fn
from eval import infer
from models import SegmentationFrameClassifer, prepare_wav2vec


@dataclass
class Segment:
    start: float
    end: float
    probs: np.array
    decimal: int = 4

    @property
    def duration(self):
        return float(round((self.end - self.start) / TARGET_SAMPLE_RATE, self.decimal))

    @property
    def offset(self):
        return float(round(self.start / TARGET_SAMPLE_RATE, self.decimal))

    @property
    def offset_plus_duration(self):
        return round(self.offset + self.duration, self.decimal)


def trim(sgm: Segment, threshold: float) -> Segment:
    """reduces the segment to between the first and last points that are above the threshold

    Args:
        sgm (Segment): a segment
        threshold (float): probability threshold

    Returns:
        Segment: new reduced segment
    """
    included_indices = np.where(sgm.probs >= threshold)[0]
    
    # return empty segment
    if not len(included_indices):
        return Segment(sgm.start, sgm.start, np.empty([0]))

    i = included_indices[0]
    j = included_indices[-1] + 1

    sgm = Segment(sgm.start + i, sgm.start + j, sgm.probs[i:j])

    return sgm


def split_and_trim(
    sgm: Segment, split_idx: int, threshold: float
) -> tuple[Segment, Segment]:
    """splits the input segment at the split_idx and then trims and returns the two resulting segments

    Args:
        sgm (Segment): input segment
        split_idx (int): index to split the input segment
        threshold (float): probability threshold

    Returns:
        tuple[Segment, Segment]: the two resulting segments
    """

    probs_a = sgm.probs[:split_idx]
    sgm_a = Segment(sgm.start, sgm.start + len(probs_a), probs_a)

    probs_b = sgm.probs[split_idx + 1 :]
    sgm_b = Segment(sgm_a.end + 1, sgm.end, probs_b)

    sgm_a = trim(sgm_a, threshold)
    sgm_b = trim(sgm_b, threshold)

    return sgm_a, sgm_b


def pdac(
    probs: np.array,
    max_segment_length: float,
    min_segment_length: float,
    threshold: float,
    not_strict: bool
) -> list[Segment]:
    """applies the probabilistic Divide-and-Conquer algorithm to split an audio
    into segments satisfying the max-segment-length and min-segment-length conditions

    Args:
        probs (np.array): the binary frame-level probabilities
            output by the segmentation-frame-classifier
        max_segment_length (float): the maximum length of a segment
        min_segment_length (float): the minimum length of a segment
        threshold (float): probability threshold
        not_strict (bool): whether segments longer than max are allowed

    Returns:
        list[Segment]: resulting segmentation
    """

    segments = []
    sgm = Segment(0, len(probs), probs)
    sgm = trim(sgm, threshold)

    def recusrive_split(sgm):
        if sgm.duration < max_segment_length:
            segments.append(sgm)
        else:
            j = 0
            sorted_indices = np.argsort(sgm.probs)
            while j < len(sorted_indices):
                split_idx = sorted_indices[j]
                split_prob = sgm.probs[split_idx]
                if not_strict and split_prob > threshold:
                    segments.append(sgm)
                    break

                sgm_a, sgm_b = split_and_trim(sgm, split_idx, threshold)
                if (
                    sgm_a.duration > min_segment_length
                    and sgm_b.duration > min_segment_length
                ):
                    recusrive_split(sgm_a)
                    recusrive_split(sgm_b)
                    break
                j += 1
            else:
                if not_strict:
                    segments.append(sgm)
                else:
                    if sgm_a.duration > min_segment_length:
                        recusrive_split(sgm_a)
                    if sgm_b.duration > min_segment_length:
                        recusrive_split(sgm_b)

    recusrive_split(sgm)

    return segments


def update_yaml_content(
    yaml_content: list[dict], segments: list[Segment], wav_name: str
) -> list[dict]:
    """extends the yaml content with the segmentation of this wav file

    Args:
        yaml_content (list[dict]): segmentation in yaml format
        segments (list[Segment]): resulting segmentation from pdac
        wav_name (str): name of the wav file

    Returns:
        list[dict]: extended segmentation in yaml format
    """
    for sgm in segments:
        yaml_content.append(
            {
                "duration": sgm.duration,
                "offset": sgm.offset,
                "rW": 0,
                "uW": 0,
                "speaker_id": "NA",
                "wav": wav_name,
            }
        )
    return yaml_content


def get_default_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.device_count() > 0:
        return torch.device("cuda:0")
    return torch.device("cpu")


def load_segmentation_models(
    path_to_checkpoint: str, device: torch.device | None = None
):
    if device is None:
        device = get_default_device()

    checkpoint = torch.load(
        path_to_checkpoint, map_location=device, weights_only=False
    )

    wav2vec_model = prepare_wav2vec(
        checkpoint["args"].model_name,
        checkpoint["args"].wav2vec_keep_layers,
        device,
    )
    sfc_model = SegmentationFrameClassifer(
        d_model=HIDDEN_SIZE,
        n_transformer_layers=checkpoint["args"].classifier_n_transformer_layers,
    ).to(device)
    sfc_model.load_state_dict(checkpoint["state_dict"])
    sfc_model.eval()

    return wav2vec_model, sfc_model, checkpoint, device


def segment_single_wav(
    path_to_wav: str | Path,
    wav2vec_model,
    sfc_model,
    device: torch.device,
    inference_batch_size: int = 12,
    inference_segment_length: int = 20,
    inference_times: int = 1,
    dac_max_segment_length: float = 18,
    dac_min_segment_length: float = 0.2,
    dac_threshold: float = 0.5,
    not_strict: bool = False,
    dataloader_num_workers: int | None = None,
    progress_callback=None,
) -> list[dict]:
    wav_path = Path(path_to_wav)
    if dataloader_num_workers is None:
        dataloader_num_workers = min(cpu_count() // 2, 4)

    dataset = FixedSegmentationDatasetNoTarget(
        wav_path, inference_segment_length, inference_times
    )
    sgm_frame_probs = None
    if progress_callback is not None:
        progress_callback(0.0)

    for inference_iteration in range(inference_times):
        dataset.fixed_length_segmentation(inference_iteration)
        dataloader = DataLoader(
            dataset,
            batch_size=inference_batch_size,
            num_workers=dataloader_num_workers,
            shuffle=False,
            drop_last=False,
            collate_fn=segm_collate_fn,
        )

        probs, _ = infer(
            wav2vec_model,
            sfc_model,
            dataloader,
            device,
        )
        if sgm_frame_probs is None:
            sgm_frame_probs = probs.copy()
        else:
            sgm_frame_probs += probs
        if progress_callback is not None:
            progress_callback((inference_iteration + 1) / inference_times * 90.0)

    sgm_frame_probs /= inference_times

    segments = pdac(
        sgm_frame_probs,
        dac_max_segment_length,
        dac_min_segment_length,
        dac_threshold,
        not_strict,
    )
    if progress_callback is not None:
        progress_callback(100.0)

    return update_yaml_content([], segments, wav_path.name)


def yaml_dump(content: list[dict]) -> str:
    return yaml.dump(content, default_flow_style=True)


def segment(args):
    device = get_default_device()
    wav2vec_model, sfc_model, _checkpoint, device = load_segmentation_models(
        args.path_to_checkpoint, device
    )

    yaml_content = []
    for wav_path in tqdm(sorted(list(Path(args.path_to_wavs).glob("*.wav")))):
        yaml_content.extend(
            segment_single_wav(
                wav_path,
                wav2vec_model,
                sfc_model,
                device,
                inference_batch_size=args.inference_batch_size,
                inference_segment_length=args.inference_segment_length,
                inference_times=args.inference_times,
                dac_max_segment_length=args.dac_max_segment_length,
                dac_min_segment_length=args.dac_min_segment_length,
                dac_threshold=args.dac_threshold,
                not_strict=args.not_strict,
            )
        )

    path_to_segmentation_yaml = Path(args.path_to_segmentation_yaml)
    path_to_segmentation_yaml.parent.mkdir(parents=True, exist_ok=True)
    with open(path_to_segmentation_yaml, "w") as f:
        f.write(yaml_dump(yaml_content))

    print(
        f"Saved SHAS segmentation with max={args.dac_max_segment_length} & "
        f"min={args.dac_min_segment_length} at {path_to_segmentation_yaml}"
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path_to_segmentation_yaml",
        "-yaml",
        type=str,
        required=True,
        help="absolute path to the yaml file to save the generated segmentation",
    )
    parser.add_argument(
        "--path_to_checkpoint",
        "-ckpt",
        type=str,
        required=True,
        help="absolute path to the audio-frame-classifier checkpoint",
    )
    parser.add_argument(
        "--path_to_wavs",
        "-wavs",
        type=str,
        help="absolute path to the directory of the wav audios to be segmented",
    )
    parser.add_argument(
        "--inference_batch_size",
        "-bs",
        type=int,
        default=12,
        help="batch size (in examples) of inference with the audio-frame-classifier",
    )
    parser.add_argument(
        "--inference_segment_length",
        "-len",
        type=int,
        default=20,
        help="segment length (in seconds) of fixed-length segmentation during inference"
        "with audio-frame-classifier",
    )
    parser.add_argument(
        "--inference_times",
        "-n",
        type=int,
        default=1,
        help="how many times to apply inference on different fixed-length segmentations"
        "of each wav",
    )
    parser.add_argument(
        "--dac_max_segment_length",
        "-max",
        type=float,
        default=18,
        help="the segmentation algorithm splits until all segments are below this value"
        "(in seconds)",
    )
    parser.add_argument(
        "--dac_min_segment_length",
        "-min",
        type=float,
        default=0.2,
        help="a split by the algorithm is carried out only if the resulting two segments"
        "are above this value (in seconds)",
    )
    parser.add_argument(
        "--dac_threshold",
        "-thr",
        type=float,
        default=0.5,
        help="after each split by the algorithm, the resulting segments are trimmed to"
        "the first and last points that corresponds to a probability above this value",
    )
    parser.add_argument(
        "--not_strict",
        action="store_true",
        help="whether segments longer than max are allowed."
        "If this argument is used, respecting the classification threshold conditions (p > thr)"
        "is more important than the length conditions (len < max)."
    )
    args = parser.parse_args()

    segment(args)
