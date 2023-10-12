import sys
from typing import Dict, Any, Tuple, Optional, Union, List
from jsonargparse import ArgumentParser, ActionConfigFile, Namespace

from .pipeline import Pipeline

ArgsType = Optional[Union[List[str], Dict[str, Any], Namespace]]

class AugmenterCLI:
    def __init__(
        self,
    ) -> None:
        self.init_parser()
        self.setup_parser()
        self.add_core_arguments_to_parser(self.parser)
        self.parse_arguments(self.parser)
        self.instantiate_classes()
        self.pipeline.run()

    def _setup_parser_kwargs(self, parser_kwargs: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        subcommand_names = self.subcommands().keys()
        main_kwargs = {k: v for k, v in parser_kwargs.items() if k not in subcommand_names}
        subparser_kwargs = {k: v for k, v in parser_kwargs.items() if k in subcommand_names}
        return main_kwargs, subparser_kwargs

    def init_parser(self, **kwargs: Any) -> ArgumentParser:
        """Method that instantiates the argument parser."""
        parser = ArgumentParser(**kwargs)
        parser.add_argument(
            "-c", "--config", action=ActionConfigFile, help="Path to a configuration file in json or yaml format."
        )
        return parser

    def setup_parser(
        self, 
        # add_subcommands: bool, main_kwargs: Dict[str, Any], subparser_kwargs: Dict[str, Any]
    ) -> None:
        """Initialize and setup the parser, subcommands, and arguments."""
        self.parser = self.init_parser()
    
    def add_core_arguments_to_parser(self, parser: ArgumentParser) -> None:
        """Adds arguments from the core classes to the parser."""
        parser.add_class_arguments(Pipeline)

    def parse_arguments(self, parser: ArgumentParser) -> None:
        """Parses command line arguments and stores it in ``self.config``."""
        self.config = parser.parse_args()

    def instantiate_classes(self) -> None:
        """Instantiates the classes and sets their attributes."""
        self.config_init = self.parser.instantiate_classes(self.config)
        self.pipeline = Pipeline(
            **{
                k: self.config_init[k]
                for k in self.config_init.keys()
                if k not in ['config']
            }
        )