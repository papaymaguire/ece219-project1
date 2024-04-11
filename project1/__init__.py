from dagster import Definitions, FilesystemIOManager, load_assets_from_modules, file_relative_path
from dagstermill import ConfigurableLocalOutputNotebookIOManager

from . import assets

all_assets = load_assets_from_modules([assets])

defs = Definitions(
    assets=all_assets,
    resources={
        "output_notebook_io_manager": ConfigurableLocalOutputNotebookIOManager(),
        "io_manager": FilesystemIOManager(base_dir=file_relative_path(__file__, "../data"))
    }
)
