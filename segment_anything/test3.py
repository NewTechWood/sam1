from segment_extractor import SegmentExtractor

extractor = SegmentExtractor()
extractor.extract_segment(
    image_path=r'D:\NewTechWood\sam1_1\segment_anything\man001.jpg',
    prompt="man",
    output_path="result.png"
)
