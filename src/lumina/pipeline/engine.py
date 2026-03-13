import logging
from typing import Any, Callable, List, Tuple
from lumina.core.image import Image

# set up a logger for this module
logger = logging.getLogger(__name__)


class Pipeline:
    # a chainable pipeline for stacking image processing steps
    # you add steps one by one and then run them all at once
    # this makes it easy to build up complex transformations

    def __init__(self) -> None:
        # each step is a (function, kwargs) tuple
        self._steps: List[Tuple[Callable[..., Image], dict[str, Any]]] = []

    def add(self, step_fn: Callable[..., Image], **kwargs: Any) -> "Pipeline":
        # add a processing step to the pipeline
        # returns self so you can chain .add().add().add()
        self._steps.append((step_fn, kwargs))
        logger.debug(f"added step: {step_fn.__name__}")
        return self

    def run(self, image: Image) -> Image:
        # execute all the steps in order, passing the image through each one
        logger.info(f"running pipeline with {len(self._steps)} steps")

        for i, (step_fn, kwargs) in enumerate(self._steps):
            step_name = step_fn.__name__
            logger.info(f"step {i + 1}/{len(self._steps)}: {step_name}")
            image = step_fn(image, **kwargs)

        logger.info("pipeline complete")
        return image

    def __len__(self) -> int:
        # how many steps are in the pipeline
        return len(self._steps)

    def __repr__(self) -> str:
        step_names = [fn.__name__ for fn, _ in self._steps]
        return f"<Pipeline steps={step_names}>"
