import sys
sys.path.append("../")
import os
import time
import gi

gi.require_version("Gst", "1.0")
import subprocess

import pyds
from common.bus_call import bus_call
from common.platform_info import PlatformInfo
from gi.repository import GLib, Gst

MUXER_BATCH_TIMEOUT_USEC = 33000

CLASS_LABELS = [
    "butter_sugar_bread", "chicken_floss_bread", "chicken_floss_sandwich",
    "cream_puff", "croissant", "donut", "muffin", "salted_egg_sponge_cake",
    "sandwich", "sponge_cake", "tiramisu"
]

CLASS_IDS = {label: idx for idx, label in enumerate(CLASS_LABELS)}


def osd_sink_pad_buffer_probe(pad, info, u_data):
    frame_number = 0
    num_rects = 0

    gst_buffer = info.get_buffer()
    if not gst_buffer:
        print("Unable to get GstBuffer")
        return

    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
    l_frame = batch_meta.frame_meta_list
    while l_frame is not None:
        try:
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
        except StopIteration:
            break

        # Intiallizing object counter with 0.
        obj_counter = {class_id: 0 for class_id in CLASS_IDS.values()}

        frame_number = frame_meta.frame_num
        num_rects = frame_meta.num_obj_meta
        l_obj = frame_meta.obj_meta_list
        while l_obj is not None:
            try:
                # Casting l_obj.data to pyds.NvDsObjectMeta
                obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
            except StopIteration:
                break
            obj_counter[obj_meta.class_id] += 1
            obj_meta.rect_params.border_color.set(
                0.0, 0.0, 1.0, 0.8
            )  # 0.8 is alpha (opacity)
            try:
                l_obj = l_obj.next
            except StopIteration:
                break
        display_meta = pyds.nvds_acquire_display_meta_from_pool(batch_meta)
        display_meta.num_labels = 1
        py_nvosd_text_params = display_meta.text_params[0]
        
        count_text = f"Frame={frame_number} | Total Objects={num_rects}"
        for class_name, class_id in CLASS_IDS.items():
            count_text += f" | {class_name}: {obj_counter[class_id]}"
        
        py_nvosd_text_params.display_text = count_text

        # Now set the offsets where the string should appear
        py_nvosd_text_params.x_offset = 10
        py_nvosd_text_params.y_offset = 12

        # Font , font-color and font-size
        py_nvosd_text_params.font_params.font_name = "Serif"
        py_nvosd_text_params.font_params.font_size = 10
        # set(red, green, blue, alpha); set to White
        py_nvosd_text_params.font_params.font_color.set(1.0, 1.0, 1.0, 1.0)

        # Text background color
        py_nvosd_text_params.set_bg_clr = 1
        # set(red, green, blue, alpha); set to Black
        py_nvosd_text_params.text_bg_clr.set(0.0, 0.0, 0.0, 1.0)
        # Using pyds.get_string() to get display_text as string
        print(pyds.get_string(py_nvosd_text_params.display_text))
        pyds.nvds_add_display_meta_to_frame(frame_meta, display_meta)
        try:
            l_frame = l_frame.next
        except StopIteration:
            break

    return Gst.PadProbeReturn.OK


def main(args):
    # Check input arguments
    if len(args) != 2:
        sys.stderr.write("usage: %s <media file or uri>\n" % args[0])
        sys.exit(1)

    platform_info = PlatformInfo()
    # Standard GStreamer initialization
    Gst.init(None)

    # Create gstreamer elements
    # Create Pipeline element that will form a connection of other elements
    print("Creating Pipeline \n ")
    pipeline = Gst.Pipeline()

    if not pipeline:
        sys.stderr.write("Unable to create Pipeline\n")

    # Source element for reading from the file
    print("Creating Source\n ")
    source = Gst.ElementFactory.make("filesrc", "file-source")
    if not source:
        sys.stderr.write("Unable to create Source\n")

    # Since the data format in the input file is elementary h264 stream,
    # we need a h264parser
    print("Creating H264Parser\n")
    h264parser = Gst.ElementFactory.make("h264parse", "h264-parser")
    if not h264parser:
        sys.stderr.write("Unable to create h264 parser\n")

    # Use nvdec_h264 for hardware accelerated decode on GPU
    print("Creating Decoder\n")
    decoder = Gst.ElementFactory.make("nvv4l2decoder", "decoder")
    decoder.set_property("enable-max-performance", 1)  # Tối ưu hiệu suất trên Jetson
    decoder.set_property("drop-frame-interval", 0)  # Không bỏ khung hình
    # decoder.set_property("num-extra-surfaces", 2)  # Cải thiện hiệu suất xử lý

    if not decoder:
        sys.stderr.write("Unable to create Nvv4l2 Decoder\n")

    # Create nvstreammux instance to form batches from one or more sources.
    streammux = Gst.ElementFactory.make("nvstreammux", "Stream-muxer")
    if not streammux:
        sys.stderr.write("Unable to create NvStreamMux\n")

    pgie = Gst.ElementFactory.make("nvinfer", "primary-inference")
    if not pgie:
        sys.stderr.write("Unable to create pgie\n")

    # Use converter to convert from NV12 to RGBA as required by nvosd
    nvvidconv = Gst.ElementFactory.make("nvvideoconvert", "converter")
    if not nvvidconv:
        sys.stderr.write("Unable to create nvvidconv\n")

    # Create OSD to draw on the converted RGBA buffer
    nvosd = Gst.ElementFactory.make("nvdsosd", "onscreendisplay")

    if not nvosd:
        sys.stderr.write("Unable to create nvosd\n")

    # Finally render the osd output
    # sink = Gst.ElementFactory.make("fakesink", "fakesink")
    # # Finally render the osd output
    # # if platform_info.is_integrated_gpu():
    # #     print("Creating nv3dsink \n")
    # #     sink = Gst.ElementFactory.make("nv3dsink", "nv3d-sink")
    # #     if not sink:
    # #         sys.stderr.write(" Unable to create nv3dsink \n")
    # # else:
    if platform_info.is_platform_aarch64():
        print("GO HERE")
        print("Creating nv3dsink \n")
        sink = Gst.ElementFactory.make("nv3dsink", "nv3d-sink")
    else:
        print("Creating EGLSink \n")
        sink = Gst.ElementFactory.make("nveglglessink", "nvvideo-renderer")
    if not sink:
        sys.stderr.write(" Unable to create egl sink \n")


    print("Playing file %s" % args[1])
    source.set_property("location", args[1])

    streammux.set_property("width", 1920)
    streammux.set_property("height", 1080)
    streammux.set_property("batched-push-timeout", MUXER_BATCH_TIMEOUT_USEC)
    streammux.set_property("batch-size", 1)
    # streammux.set_property("nvbuf-memory-type", 2)

    pgie.set_property("config-file-path", "../../configs/config_primary_yolov11.txt")

    # nvvidconv.set_property("nvbuf-memory-type", 2)

    # decoder.set_property("cudadec-memtype", 2)

    print("Adding elements to Pipeline \n")
    pipeline.add(source)
    pipeline.add(h264parser)
    pipeline.add(decoder)
    pipeline.add(streammux)
    pipeline.add(pgie)
    pipeline.add(nvvidconv)
    pipeline.add(nvosd)
    pipeline.add(sink)

    print("Linking elements in the Pipeline \n")
    source.link(h264parser)
    h264parser.link(decoder)

    sinkpad = streammux.get_request_pad("sink_0")
    if not sinkpad:
        sys.stderr.write("Unable to get the sink pad of streammux\n")
    srcpad = decoder.get_static_pad("src")
    if not srcpad:
        sys.stderr.write("Unable to get source pad of decoder\n")
    srcpad.link(sinkpad)
    streammux.link(pgie)
    pgie.link(nvvidconv)
    nvvidconv.link(nvosd)
    nvosd.link(sink)

    # create an event loop and feed gstreamer bus messages to it
    loop = GLib.MainLoop()
    bus = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect("message", bus_call, loop)

    osdsinkpad = nvosd.get_static_pad("sink")
    if not osdsinkpad:
        sys.stderr.write("Unable to get sink pad of nvosd\n")

    osdsinkpad.add_probe(Gst.PadProbeType.BUFFER, osd_sink_pad_buffer_probe, 0)

    # start play back and listen to events
    print("Starting pipeline \n")
    pipeline.set_state(Gst.State.PLAYING)
    try:
        loop.run()
    except Exception as e:
        raise e
    # cleanup
    pipeline.set_state(Gst.State.NULL)


if __name__ == "__main__":
    sys.exit(main(sys.argv))
