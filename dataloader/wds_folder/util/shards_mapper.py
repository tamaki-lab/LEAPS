import abc
import json
import pickle

import webdataset as wds


class WdsPipelineInterface(metaclass=abc.ABCMeta):
    def __init__(self, dataset_name: str):
        self.dataset_name = dataset_name

        self.last_opened_video_filename = None
        self.last_opened_video_fps = None
        self.last_opened_video_category = None

    def get_label(self, batch, label_key: list[str], category_key: list[str] | None = None) -> int | None:
        l_key = next((l_k for l_k in label_key if l_k in batch), None)
        if l_key is None:
            return None
        label = batch[l_key]
        return label

    def get_category(self, batch, category_key: list[str]) -> str:
        c_key = next((c_k for c_k in category_key if c_k in batch), None)
        if c_key is None:
            return None
        category = batch[c_key]
        self.last_opened_video_category = category
        return category

    def store_fps(self, fps: float) -> float:
        if fps is None or fps <= 0:
            # If FPS is not provided or invalid, use a default value
            return 30.0
        self.last_opened_video_fps = fps
        return fps

    @abc.abstractmethod
    def _get_decode_callbacks(self) -> list:
        raise NotImplementedError()

    @abc.abstractmethod
    def _get_to_tuple_members(self) -> list:
        raise NotImplementedError()

    @abc.abstractmethod
    def _get_map_tuple_label_category_callbacks(self) -> list:
        raise NotImplementedError()

    @abc.abstractmethod
    def _get_map_tuple_content_callbacks(self, decode_video) -> list:
        raise NotImplementedError()

    def _get_dataset_pipeline_members(self, video_decoder) -> tuple[list, list, list, list]:
        return (
            self._get_decode_callbacks(),
            self._get_to_tuple_members(),
            self._get_map_tuple_label_category_callbacks(),
            self._get_map_tuple_content_callbacks(video_decoder),
        )

    def __call__(self, dataset: wds.WebDataset, video_decoder) -> wds.WebDataset:
        d_params, t_params, m_label_params, m_content_params = self._get_dataset_pipeline_members(video_decoder)
        ds = dataset.decode(*d_params).to_tuple(*t_params).map_tuple(*m_label_params)
        ds = ds.map_tuple(*m_content_params)
        return ds

    @abc.abstractmethod
    def collate_fn(self, batch) -> dict:
        raise NotImplementedError()


class StandardWdsPipeline(WdsPipelineInterface):
    def _get_decode_callbacks(self) -> list:
        callbacks = [
            wds.handle_extension("video.pickle", pickle.loads),
            wds.handle_extension("stats.json", json.loads),
        ]
        return callbacks

    def _get_to_tuple_members(self) -> list:
        members = [
            "video.pickle",
            "stats.json",
            "stats.json",
            "stats.json",
            "stats.json",
            "stats.json",
            "stats.json",
            "stats.json",
            "stats.json",
            "stats.json",
            "stats.json",
            "stats.json",
            "stats.json",
        ]
        return members

    def _get_map_tuple_label_category_callbacks(self) -> list:
        l_key = ["label"]
        c_key = ["category"]
        callbacks = [
            lambda x: x,
            lambda x: self.get_label(x, l_key, c_key),  # label
            lambda x: self.get_category(x, c_key),  # label text
            lambda x: x["filename"],
            lambda x: self.store_fps(x["fps"]),
            lambda x: self.get_label(x, ["bg_label"], ["bg_category"]),
            lambda x: self.get_category(x, ["bg_category"]),
            lambda x: self.get_category(x, ["bg_filename"]),
            lambda x: self.get_label(x, ["verb_label"]),
            lambda x: self.get_label(x, ["noun_label"]),
            lambda x: self.get_category(x, ["verb_category"]),
            lambda x: self.get_category(x, ["noun_category"]),
        ]
        return callbacks

    def _get_map_tuple_content_callbacks(self, decode_video) -> list:
        callbacks = [
            lambda x: decode_video.video_decoder(x, source_fps=self.last_opened_video_fps),
            lambda x: x,  # label
            lambda x: x,  # label text
            lambda x: x,  # filename
            lambda x: x,  # fps
            lambda x: x,  # background label
            lambda x: x,  # background label text
            lambda x: x,  # background filename
            lambda x: x,  # verb label
            lambda x: x,  # noun label
            lambda x: x,  # verb category
            lambda x: x,  # noun category
        ]
        return callbacks

    def collate_fn(self, batch):
        ret = {
            "video": batch[0],
            "label": batch[1],
            "label_text": batch[2],
            "filename": batch[3],
            "fps": batch[4],
            "bg_label": batch[5],
            "bg_label_text": batch[6],
            "bg_filename": batch[7],
            "verb_label": batch[8],
            "noun_label": batch[9],
            "verb_label_text": batch[10],
            "noun_label_text": batch[11],
            "dataset_name": self.dataset_name,
        }
        return ret
