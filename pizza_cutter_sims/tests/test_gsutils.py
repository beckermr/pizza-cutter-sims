import galsim

from pizza_cutter_sims.gsutils import build_gsobject


def test_build_gsobject_const():
    config = {
        "type": "Exponential",
        "half_light_radius": 0.5
    }
    obj = build_gsobject(config=config, kind='gal')
    assert isinstance(obj, galsim.Exponential)
    assert obj.half_light_radius == 0.5


def test_build_gsobject_eval():
    config = {
        "type": "Exponential",
        "half_light_radius": "$blah * 0.05"
    }
    eval_variables = {
        "fblah": 10,
    }
    obj = build_gsobject(config=config, kind='gal', eval_variables=eval_variables)
    assert isinstance(obj, galsim.Exponential)
    assert obj.half_light_radius == 0.5


def test_build_gsobject_rng_eval():
    config = {
        "type": "Exponential",
        "half_light_radius": {
            "type": "Eval",
            "str": "blah * ran",
            "fran": {"type": "Random", "min": 0.1, "max": 0.2},
        },
    }
    eval_variables = {
        "fblah": 10,
    }
    obj = build_gsobject(
        config=config,
        kind='gal',
        eval_variables=eval_variables,
        rng=galsim.BaseDeviate(42)
    )
    assert isinstance(obj, galsim.Exponential)
    assert obj.half_light_radius >= 1
    assert obj.half_light_radius <= 2

    obj_again = build_gsobject(
        config=config,
        kind='gal',
        eval_variables=eval_variables,
        rng=galsim.BaseDeviate(42)
    )
    assert isinstance(obj_again, galsim.Exponential)
    assert obj.half_light_radius == obj_again.half_light_radius
