import os
import re
import cv2
import torch
import numpy as np
from tqdm import tqdm
from transformers import VideoMAEImageProcessor, VideoMAEModel
import torch.nn.functional as F

torch.set_float32_matmul_precision("high")


class PretrainedVideoExtractor:
    def __init__(self):
        model_name = "MCG-NJU/videomae-base"

        print("Loading VideoMAE...")

        self.processor = VideoMAEImageProcessor.from_pretrained(
            model_name
        )

        self.model = (
            VideoMAEModel.from_pretrained(model_name)
            .cuda()
            .eval()
        )

    @torch.no_grad()
    def extract(self, video_frames):
        """
        video_frames:
            List[np.ndarray]
            length = T
            each frame = (H,W,C)
            RGB uint8
        """

        inputs = self.processor(
            [video_frames],
            return_tensors="pt"
        ).to("cuda")

        outputs = self.model(**inputs)

        feature = outputs.last_hidden_state.mean(dim=1)

        feature = F.normalize(feature, dim=-1)

        return feature.squeeze(0).cpu().numpy()


def load_video(video_path, num_frames=16):
    cap = cv2.VideoCapture(video_path)

    frames = []

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        frames.append(frame)

    cap.release()

    if len(frames) == 0:
        raise RuntimeError(f"Cannot read video: {video_path}")

    # 均匀采样16帧
    indices = np.linspace(
        0,
        len(frames) - 1,
        num_frames,
        dtype=int
    )

    sampled_frames = [
        frames[i]
        for i in indices
    ]

    return sampled_frames


def get_video_id(filename):
    """
    786_Select0001-0100.mp4
    ->
    786
    """

    match = re.match(r"(\d+)_Select", filename)

    if match:
        return match.group(1)

    return os.path.splitext(filename)[0]


def main():

    video_dirs = [
        "selected_segmented_HumanML3D_bvh_train_mp4",
        "selected_segmented_HumanML3D_bvh_val_mp4",
        "selected_segmented_HumanML3D_bvh_test_mp4",
    ]

    output_dir = (
        "EEG_data/"
        "videomae_global_features"
    )

    os.makedirs(output_dir, exist_ok=True)

    extractor = PretrainedVideoExtractor()

    all_videos = []

    for folder in video_dirs:
        for file in os.listdir(folder):

            if file.endswith(".mp4"):
                all_videos.append(
                    os.path.join(folder, file)
                )

    print("Total videos:", len(all_videos))

    for video_path in tqdm(all_videos):

        filename = os.path.basename(video_path)

        video_id = get_video_id(filename)

        save_path = os.path.join(
            output_dir,
            f"{video_id}.npy"
        )

        if os.path.exists(save_path):
            continue

        try:

            frames = load_video(
                video_path,
                num_frames=16
            )

            feature = extractor.extract(frames)

            np.save(
                save_path,
                feature.astype(np.float32)
            )

        except Exception as e:

            print(
                f"Failed: {video_path}"
            )

            print(e)

        # break


if __name__ == "__main__":
    main()