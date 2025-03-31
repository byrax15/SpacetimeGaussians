

if __name__ == "__main__":
    from tap import Tap, tapify
    from pathlib import Path
    import re

    class ArgumentParser(Tap):
        videosdir: Path
        prior_data_fmt: str = "txt"
        imageext: str = "png"
        dryrun: bool = False
        startframe: int = 0
        endframe: int = 50

    class SparsePriorModel:
        def __init__(self, videosdir: Path, prior_data_fmt: str):
            self.prior = videosdir / "prior"
            self.cameras = self.prior / f"cameras.{prior_data_fmt}"
            self.images = self.prior / f"images.{prior_data_fmt}"
            self.points3D = self.prior / f"points3D.{prior_data_fmt}"

            assert self.prior.exists()
            assert self.cameras.exists()
            assert self.images.exists()
            assert self.points3D.exists()

        def __repr__(self):
            return f"""SparseModel(
    prior = {self.prior},
    cameras = {self.cameras},
    images = {self.images},
    points3D = {self.points3D}
)"""

    def colmap_input_dir(point_dir: Path, frame: int) -> Path:
        return point_dir / f"colmap_{frame}" / "input"

    def colmap_input_image_path(point_dir: Path, frame: int, cam_num: int, imageext: str) -> Path:
        return colmap_input_dir(point_dir, frame) / f"{cam_num}.{imageext}"

    args = ArgumentParser().parse_args()
    prior_data = tapify(SparsePriorModel, known_only=True)
    frames_dir = args.videosdir / "frames"
    assert frames_dir.exists()
    point_dir = args.videosdir / "point"
    assert point_dir.exists()

    for f in range(args.startframe, args.endframe):
        colmap_input_dir(point_dir, f).mkdir(parents=True, exist_ok=True)

    cam_pattern = re.compile(r"cam(\d+)")
    frame_pattern = re.compile(r"(\d+)")
    for cam in frames_dir.glob("cam*"):
        cam_num = int(cam_pattern.match(cam.name)[1])
        for frame in cam.glob(f"*.{args.imageext}"):
            frame_num = int(frame_pattern.match(frame.name)[1])
            link_path = colmap_input_image_path(
                point_dir, frame_num, cam_num, args.imageext)
            if args.dryrun:
                print(f"'{link_path}'.symlink_to('{frame.absolute()}')")
            else:
                link_path.symlink_to(frame.absolute())

