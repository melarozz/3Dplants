'''plant-cpd: CPD-based registration of 3D plant models.'''
from plant_3d.infrastructure.cpd.config import NonrigidCPDConfig, PipelineConfig, PreprocessConfig, RigidCPDConfig
from plant_3d.infrastructure.cpd.pipeline import register
from plant_3d import __version__
__all__ = [
    'register',
    'PipelineConfig',
    'PreprocessConfig',
    'RigidCPDConfig',
    'NonrigidCPDConfig',
    '__version__']
