"""Teacher model synthesis utilities"""

from .teacher_query import TeacherModelClient, TeacherResponse, create_code_generation_prompt
from .generate_traces import generate_synthetic_data, load_synthetic_data

__all__ = [
    "TeacherModelClient",
    "TeacherResponse",
    "create_code_generation_prompt",
    "generate_synthetic_data",
    "load_synthetic_data",
]
