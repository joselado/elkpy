from elkpy.inputfile import InputFile


def test_render_basic_blocks():
    f = InputFile()
    f.add_block("tasks", [0, 1])
    f.add_block("avec", [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)])
    f.add_block("sppath", ["/path/to/species/"])
    f.add_block("spinpol", [True])
    text = f.render()

    lines = [line for line in text.split("\n")]
    assert "tasks" in lines
    assert "0" in lines
    assert "1" in lines
    assert "sppath" in lines
    assert "'/path/to/species/'" in lines
    assert "spinpol" in lines
    assert ".true." in lines


def test_blocks_terminated_by_blank_line():
    f = InputFile()
    f.add_block("tasks", [0])
    f.add_block("ngridk", [(2, 2, 2)])
    text = f.render()
    assert "tasks\n0\n\nngridk\n2  2  2\n\n" == text
