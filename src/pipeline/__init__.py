"""Pipeline: end-to-end AIGC video production pipeline."""
from .schema import Scene, SceneType, TextOverlay, VideoProject, Transition
from .script_writer import ScriptWriter
from .generator import AssetProducer
from .composer import Composer
