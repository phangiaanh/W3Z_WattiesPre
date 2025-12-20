from setuptools import setup, find_packages

print('Found packages:', find_packages())
setup(
    description='Watties as a package',
    name='Watties',
    packages=find_packages(),
    install_requires=[
        'gdown',
        'opencv-python',
        'pyrender',
        'pytorch-lightning',
        'scikit-image',
        'smplx==0.1.28',
        'yacs',
        'detectron2 @ file:///tmp/detectron2-0.6-cp312-cp312-linux_x86_64.whl',
        'chumpy @ file:///tmp/chumpy-0.71-py3-none-any.whl',
        'mmcv==1.3.9',
        'timm',
        'einops',
    ],
    extras_require={
        'all': [
            'hydra-core',
            'hydra-submitit-launcher',
            'hydra-colorlog',
            'pyrootutils',
            'rich',
        ],
    },
)
