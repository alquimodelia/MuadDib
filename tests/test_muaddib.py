import os
import pathlib

import pytest

from muaddib.muaddib import ProjectFolder
from tests.test_arrakis.test.data.definitions import ALL_DATA_MANAGERS

TESTS_FOLDER = pathlib.Path(os.path.dirname(os.path.abspath(__file__)))
TEST_ARRAKIS_ROOT = TESTS_FOLDER.joinpath("test_arrakis")

ProjectFolderArgs = {
    "name": "tests",
    "root_folder": TEST_ARRAKIS_ROOT,
    "target_variables": ["target_variable1", "target_variable2"],
}


@pytest.mark.parametrize("ProjectFolderArgs", [ProjectFolderArgs])
def test_ProjectFolder(ProjectFolderArgs):
    test_Project = ProjectFolder(**ProjectFolderArgs)


@pytest.mark.parametrize("DataMan", ALL_DATA_MANAGERS)
def test_DataHandle(DataMan):
    dataman = DataMan()
