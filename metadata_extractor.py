"""
ComfyUI Metadata Extractor - Improved Version
Extracts metadata from PNG images and MP4/video files
Now properly extracts: prompts, seed, steps, cfg, sampler, scheduler
"""

import json
import os
import subprocess
import folder_paths

try:
    from PIL import Image
except ImportError:
    import PIL.Image as Image


class MetadataExtractorImproved:
    """
    Extracts metadata from ComfyUI-generated files
    Properly parses seed, steps, cfg from the prompt structure
    """

    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.input_dir = folder_paths.get_input_directory()

    @classmethod
    def INPUT_TYPES(cls):
        input_dir = folder_paths.get_input_directory()
        files = []

        try:
            files = [f for f in os.listdir(input_dir) 
                    if os.path.isfile(os.path.join(input_dir, f)) and 
                    f.lower().endswith(('.png', '.jpg', '.jpeg', '.mp4', '.avi', '.mov', '.mkv', '.webm'))]
        except:
            pass

        return {
            "required": {},
            "optional": {
                "video": ("IMAGE",),
                "file_path": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "Full path to file"
                }),
                "filename": ([""] + sorted(files), {
                    "default": ""
                }),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "INT", "INT", "FLOAT", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("positive_prompt", "negative_prompt", "seed", "steps", "cfg", "sampler", "scheduler", "metadata_json", "file_info")
    FUNCTION = "extract"
    CATEGORY = "utils/metadata"
    OUTPUT_NODE = False

    def find_video_file_in_workflow(self, video_tensor):
        """Try to find source video file"""
        possible_locations = [self.input_dir, self.output_dir]
        video_files = []

        for location in possible_locations:
            if not os.path.exists(location):
                continue
            try:
                for f in os.listdir(location):
                    if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.webm')):
                        full_path = os.path.join(location, f)
                        video_files.append((full_path, os.path.getmtime(full_path)))
            except:
                pass

        video_files.sort(key=lambda x: x[1], reverse=True)
        if video_files:
            print(f"[Metadata] 💡 Found: {os.path.basename(video_files[0][0])}")
            return video_files[0][0]
        return None

    def extract_video_metadata(self, file_path):
        """Extract metadata from video files using ffprobe"""
        try:
            cmd = ['ffprobe', '-v', 'quiet', '-print_format', 'json',
                   '-show_format', '-show_streams', file_path]

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            if result.returncode != 0:
                return None

            video_info = json.loads(result.stdout)
            tags = video_info.get('format', {}).get('tags', {})

            print(f"[Metadata] Video tags: {list(tags.keys())}")

            metadata = {}
            metadata_fields = ['comment', 'Comment', 'description', 'Description', 
                             'workflow', 'Workflow', 'prompt', 'Prompt']

            for field in metadata_fields:
                if field in tags:
                    try:
                        parsed = json.loads(tags[field])
                        metadata[field.lower()] = parsed
                        print(f"[Metadata] ✅ Found '{field}' tag")
                    except:
                        metadata[field.lower()] = tags[field]

            # Get video properties
            for stream in video_info.get('streams', []):
                if stream.get('codec_type') == 'video':
                    metadata['video_width'] = stream.get('width', 0)
                    metadata['video_height'] = stream.get('height', 0)
                    fps_str = stream.get('r_frame_rate', '0/1')
                    try:
                        fps_num, fps_den = map(int, fps_str.split('/'))
                        metadata['video_fps'] = fps_num / fps_den if fps_den else 0
                    except:
                        metadata['video_fps'] = 0
                    metadata['video_duration'] = float(stream.get('duration', 0))
                    break

            return metadata if metadata else None

        except FileNotFoundError:
            print("[Metadata] ❌ ffprobe not found! Install ffmpeg")
            return None
        except Exception as e:
            print(f"[Metadata] Error: {e}")
            return None

    def extract_png_metadata(self, file_path):
        """Extract metadata from PNG files"""
        try:
            img = Image.open(file_path)
            metadata = {}

            if hasattr(img, 'info'):
                if 'workflow' in img.info:
                    try:
                        metadata['workflow'] = json.loads(img.info['workflow'])
                    except:
                        pass
                if 'prompt' in img.info:
                    try:
                        metadata['prompt'] = json.loads(img.info['prompt'])
                    except:
                        pass

            return metadata if metadata else None
        except Exception as e:
            print(f"[Metadata] Error reading PNG: {e}")
            return None

    def parse_comfyui_metadata(self, metadata):
        """Parse metadata and extract ALL parameters"""
        result = {
            'positive_prompt': "",
            'negative_prompt': "",
            'seed': 0,
            'steps': 0,
            'cfg': 0.0,
            'sampler': "",
            'scheduler': "",
            'metadata_json': "{}",
            'file_info': ""
        }

        try:
            # Get prompt data (node-based structure)
            prompt_data = None

            if 'prompt' in metadata:
                prompt_data = metadata['prompt']
            elif 'comment' in metadata and isinstance(metadata['comment'], dict):
                prompt_data = metadata['comment']

            if not prompt_data:
                result['metadata_json'] = json.dumps(metadata, indent=2)
                return result

            print(f"[Metadata] Parsing {len(prompt_data)} nodes...")

            # Track what we've found
            positive_found = False
            negative_found = False

            # Iterate through all nodes
            for node_id, node_data in prompt_data.items():
                if not isinstance(node_data, dict):
                    continue

                inputs = node_data.get('inputs', {})
                class_type = node_data.get('class_type', '')

                # Extract prompts
                if class_type in ['CLIPTextEncode', 'CLIPTextEncodeSDXL', 'CLIPTextEncodeFlux']:
                    text_content = inputs.get('text', '')
                    if text_content:
                        if not positive_found:
                            result['positive_prompt'] = text_content
                            positive_found = True
                            print(f"[Metadata] ✅ Positive prompt")
                        elif not negative_found:
                            result['negative_prompt'] = text_content
                            negative_found = True
                            print(f"[Metadata] ✅ Negative prompt")

                # Extract seed from RandomNoise node
                if class_type == 'RandomNoise':
                    if 'noise_seed' in inputs:
                        result['seed'] = int(inputs['noise_seed'])
                        print(f"[Metadata] ✅ Seed: {result['seed']}")
                    elif isinstance(inputs.get('noise_seed'), list):
                        # Handle array format: ["get", ["123", 0]]
                        try:
                            result['seed'] = int(inputs['noise_seed'][0])
                        except:
                            pass

                # Extract sampler
                if class_type == 'KSamplerSelect':
                    sampler_name = inputs.get('sampler_name', '')
                    if sampler_name:
                        result['sampler'] = sampler_name
                        print(f"[Metadata] ✅ Sampler: {sampler_name}")

                # Extract steps and cfg from scheduler nodes
                if class_type in ['LTXVScheduler', 'BasicScheduler', 'KScheduler']:
                    if 'steps' in inputs:
                        result['steps'] = int(inputs['steps'])
                        print(f"[Metadata] ✅ Steps: {result['steps']}")

                # Extract CFG from guider nodes
                if class_type in ['CFGGuider', 'DualCFGGuider']:
                    if 'cfg' in inputs:
                        result['cfg'] = float(inputs['cfg'])
                        print(f"[Metadata] ✅ CFG: {result['cfg']}")

            # Store full metadata
            result['metadata_json'] = json.dumps(metadata, indent=2)

            # Create file info
            info_parts = []
            if 'video_width' in metadata:
                info_parts.append(f"{metadata['video_width']}x{metadata['video_height']}")
            if 'video_fps' in metadata:
                info_parts.append(f"{metadata['video_fps']:.2f} FPS")
            if 'video_duration' in metadata:
                info_parts.append(f"{metadata['video_duration']:.2f}s")

            result['file_info'] = " | ".join(info_parts) if info_parts else "PNG Image"

        except Exception as e:
            print(f"[Metadata] Error parsing: {e}")
            import traceback
            traceback.print_exc()

        return result

    def extract(self, video=None, file_path="", filename=""):
        """Main extraction function"""

        actual_path = None

        if file_path and os.path.exists(file_path):
            actual_path = file_path
        elif filename:
            actual_path = os.path.join(self.input_dir, filename)
        elif video is not None:
            actual_path = self.find_video_file_in_workflow(video)

        if not actual_path or not os.path.exists(actual_path):
            return ("", "", 0, 0, 0.0, "", "", "{}", "File not found")

        print(f"[Metadata] 📂 Processing: {os.path.basename(actual_path)}")

        ext = os.path.splitext(actual_path)[1].lower()

        metadata = None
        if ext in ['.png', '.jpg', '.jpeg']:
            metadata = self.extract_png_metadata(actual_path)
        elif ext in ['.mp4', '.avi', '.mov', '.mkv', '.webm']:
            metadata = self.extract_video_metadata(actual_path)
        else:
            return ("", "", 0, 0, 0.0, "", "", "{}", f"Unsupported: {ext}")

        if not metadata:
            return ("", "", 0, 0, 0.0, "", "", "{}", "No metadata found")

        result = self.parse_comfyui_metadata(metadata)

        return (
            result['positive_prompt'],
            result['negative_prompt'],
            result['seed'],
            result['steps'],
            result['cfg'],
            result['sampler'],
            result['scheduler'],
            result['metadata_json'],
            result['file_info']
        )


NODE_CLASS_MAPPINGS = {
    "MetadataExtractorImproved": MetadataExtractorImproved
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MetadataExtractorImproved": "📹 Extract Metadata (PNG/Video)"
}
