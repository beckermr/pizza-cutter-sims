from ..run_utils import cut_nones


def test_cut_nones():
    pres = [1, None, 2, None, 4]
    mres = [None, 11, 12, None, 14]

    pres, mres = cut_nones(pres, mres)

    assert pres == [2, 4]
    assert mres == [12, 14]
