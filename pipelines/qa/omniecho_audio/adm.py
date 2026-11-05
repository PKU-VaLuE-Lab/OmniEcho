from __future__ import annotations

from pathlib import Path
import xml.etree.ElementTree as ET
from xml.dom import minidom

from .schema import AudioObject, Scene, load_scene

ADM_NS = "urn:ebu:metadata-schema:ebuCore_2014"
ET.register_namespace("", ADM_NS)


def _adm_id(prefix: str, index: int) -> str:
    return f"{prefix}_{index:04d}"


def _trajectory_text(source: AudioObject) -> str:
    points = []
    for point in source.trajectory:
        points.append(
            f"t={point.time:.6f},azimuth={point.azimuth:.6f},"
            f"elevation={point.elevation:.6f},distance={point.distance:.6f},gain_db={point.gain_db:.6f}"
        )
    return ";".join(points)


def build_adm_xml(scene: Scene) -> ET.ElementTree:
    root = ET.Element(f"{{{ADM_NS}}}ebuCoreMain", {"schemaVersion": "1.10"})
    core_metadata = ET.SubElement(root, f"{{{ADM_NS}}}coreMetadata")
    format_node = ET.SubElement(core_metadata, f"{{{ADM_NS}}}format")
    audio_format = ET.SubElement(format_node, f"{{{ADM_NS}}}audioFormatExtended")

    programme_id = "APR_0001"
    content_id = "ACO_0001"
    programme = ET.SubElement(
        audio_format,
        "audioProgramme",
        {"audioProgrammeID": programme_id, "audioProgrammeName": scene.scene_id},
    )
    ET.SubElement(programme, "audioContentIDRef").text = content_id
    content = ET.SubElement(
        audio_format,
        "audioContent",
        {"audioContentID": content_id, "audioContentName": f"{scene.scene_id}_objects"},
    )

    for index, source in enumerate(scene.objects, start=1):
        object_id = _adm_id("AO", index)
        pack_id = _adm_id("AP", index)
        channel_id = _adm_id("AC", index)
        track_uid = f"ATU_{index:08d}"
        track_format_id = _adm_id("AT", index)
        stream_format_id = _adm_id("AS", index)
        content_ref = ET.SubElement(content, "audioObjectIDRef")
        content_ref.text = object_id

        audio_object = ET.SubElement(
            audio_format,
            "audioObject",
            {
                "audioObjectID": object_id,
                "audioObjectName": source.object_id,
                "start": f"00:00:{source.start_time:06.3f}",
            },
        )
        if source.end_time is not None:
            audio_object.set("duration", f"00:00:{max(0.0, source.end_time - source.start_time):06.3f}")
        ET.SubElement(audio_object, "audioPackFormatIDRef").text = pack_id
        ET.SubElement(audio_object, "audioTrackUIDRef").text = track_uid

        pack = ET.SubElement(
            audio_format,
            "audioPackFormat",
            {"audioPackFormatID": pack_id, "audioPackFormatName": f"{source.object_id}_pack", "typeLabel": "0003", "typeDefinition": "Objects"},
        )
        ET.SubElement(pack, "audioChannelFormatIDRef").text = channel_id

        channel = ET.SubElement(
            audio_format,
            "audioChannelFormat",
            {"audioChannelFormatID": channel_id, "audioChannelFormatName": f"{source.object_id}_channel", "typeLabel": "0003", "typeDefinition": "Objects"},
        )
        block = ET.SubElement(channel, "audioBlockFormat", {"audioBlockFormatID": f"{channel_id}_00000001", "rtime": "00:00:00.000"})
        first = source.trajectory[0]
        ET.SubElement(block, "cartesian").text = "0"
        ET.SubElement(block, "position", {"coordinate": "azimuth"}).text = f"{first.azimuth:.6f}"
        ET.SubElement(block, "position", {"coordinate": "elevation"}).text = f"{first.elevation:.6f}"
        ET.SubElement(block, "position", {"coordinate": "distance"}).text = f"{first.distance:.6f}"
        ET.SubElement(block, "gain").text = f"{source.gain_db:.6f}"
        ET.SubElement(block, "width").text = f"{source.spread_deg:.6f}"
        ET.SubElement(block, "objectDivergence").text = "0"
        ET.SubElement(block, "jumpPosition", {"interpolationLength": "00:00:00.000"}).text = "0"
        ET.SubElement(block, "trajectoryMetadata").text = _trajectory_text(source)
        ET.SubElement(block, "sourcePath").text = str(source.wav_path)

        track_uid_node = ET.SubElement(audio_format, "audioTrackUID", {"UID": track_uid})
        ET.SubElement(track_uid_node, "audioTrackFormatIDRef").text = track_format_id
        ET.SubElement(track_uid_node, "audioPackFormatIDRef").text = pack_id

        track_format = ET.SubElement(
            audio_format,
            "audioTrackFormat",
            {"audioTrackFormatID": track_format_id, "audioTrackFormatName": f"{source.object_id}_track", "formatLabel": "0001", "formatDefinition": "PCM"},
        )
        ET.SubElement(track_format, "audioStreamFormatIDRef").text = stream_format_id
        stream_format = ET.SubElement(
            audio_format,
            "audioStreamFormat",
            {"audioStreamFormatID": stream_format_id, "audioStreamFormatName": f"{source.object_id}_stream", "formatLabel": "0001", "formatDefinition": "PCM"},
        )
        ET.SubElement(stream_format, "audioChannelFormatIDRef").text = channel_id
        ET.SubElement(stream_format, "audioPackFormatIDRef").text = pack_id

    return ET.ElementTree(root)


def write_adm_xml(scene_path: str | Path, output_path: str | Path) -> Path:
    scene = load_scene(scene_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tree = build_adm_xml(scene)
    rough = ET.tostring(tree.getroot(), encoding="utf-8")
    pretty = minidom.parseString(rough).toprettyxml(indent="  ", encoding="utf-8")
    output_path.write_bytes(pretty)
    return output_path
