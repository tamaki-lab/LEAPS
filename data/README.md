# WebDataset for video data

Applicable webdataset json structure for video data is as follows. The key string is used for sharding and indexing, and the video data is stored in a pickle file containing a list of JPEG bytes and a list of frame timestamps. The stats.json file contains metadata about the video, including its dimensions, frame rate, duration, and labels for the video and its background, verb, and noun categories if applicable.

```py
{
    '**key**': key_str,
    'video.pickle': (jpg_byte_list, frame_sec_list),
    'stats.json': {
        '__key__': key_str,
        'video_id': video_file_path.stem,
        'filename': video_file_path.name,
        'category': category_name,
        'label': label,
        'width': stream.codec_context.width,
        'height': stream.codec_context.height,
        'fps': float(stream.base_rate),
        'n_frames': n_frames,
        'duration': float(container.duration) / av.time_base,
        'timestamps': frame_sec_list,
        'bg_label': bg_label | None,
        'bg_category': bg_category | None,
        'bg_filename': bg_filename | None,
        'verb_label': verb_label | None,
        'verb_category': verb_category | None,
        'noun_label': noun_label | None,
        'noun_category': noun_category | None,
    }
}
```
