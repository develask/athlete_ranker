import unittest
from types import SimpleNamespace

from performance_groups import prepare_performance_groups
from rating_engine import (
    FIELD_SIZE_REFERENCE,
    calculate_team_base_deltas,
    effective_k,
    field_size_multiplier,
    process_performance_group,
)


def entry(athlete_ids, time_seconds):
    return SimpleNamespace(athlete_ids=tuple(athlete_ids), time_seconds=time_seconds)


class FieldSizeKTests(unittest.TestCase):
    def test_alpha_zero_exactly_reproduces_fixed_k(self):
        group = SimpleNamespace(entries=[entry(["a"], 60.0), entry(["b"], 70.0)])
        deltas = calculate_team_base_deltas(
            group,
            [1500.0, 1500.0],
            k=100.0,
            field_size_alpha=0.0,
        )
        self.assertEqual(deltas, [50.0, -50.0])

    def test_reference_and_direction(self):
        self.assertEqual(field_size_multiplier(FIELD_SIZE_REFERENCE), 1.0)
        self.assertLess(field_size_multiplier(10), 1.0)
        self.assertGreater(field_size_multiplier(40), 1.0)

    def test_clipping(self):
        self.assertEqual(field_size_multiplier(1, alpha=1.0), 0.70)
        self.assertEqual(field_size_multiplier(1000, alpha=1.0), 1.75)

    def test_team_size_uses_boats_not_paddlers(self):
        group = SimpleNamespace(
            entries=[
                entry(["a1", "a2", "a3", "a4"], 60.0),
                entry(["b1", "b2", "b3", "b4"], 70.0),
            ]
        )
        deltas = calculate_team_base_deltas(
            group,
            [1500.0, 1500.0],
            k=100.0,
            field_size_alpha=0.25,
        )
        expected_delta = effective_k(100.0, n_entries=2) * 0.5
        self.assertAlmostEqual(deltas[0], expected_delta)
        self.assertAlmostEqual(deltas[1], -expected_delta)

    def test_cleaned_duplicate_does_not_increase_field_size(self):
        prueba = {
            "prueba_id": "p1",
            "prueba_fase": "final",
            "tipo": "aguas_tranquilas",
            "embarcacion_tipo": "kayak",
            "embarcacion_num": 1,
            "distancia_exacta": 500,
            "length_class": "super_sprint",
            "results": [
                {"lista_palista_id": ["a"], "tiempo": "00:01:00.00"},
                {"lista_palista_id": ["a"], "tiempo": "00:01:00.20"},
                {"lista_palista_id": ["b"], "tiempo": "00:01:10.00"},
            ],
        }
        groups, _ = prepare_performance_groups(
            [{"regata_id": "r1", "fecha": "2025-01-01", "pruebas": [prueba]}]
        )
        group = groups[0]
        self.assertEqual(len(group.entries), 2)

        result = process_performance_group({}, group, k=100.0)
        self.assertEqual(result.n_entries, 2)
        self.assertEqual(
            result.field_size_multiplier,
            field_size_multiplier(2),
        )

    def test_scaling_changes_updates_not_pre_update_metrics(self):
        prueba = {
            "prueba_id": "p1",
            "prueba_fase": "final",
            "tipo": "aguas_tranquilas",
            "embarcacion_tipo": "kayak",
            "embarcacion_num": 1,
            "distancia_exacta": 500,
            "length_class": "super_sprint",
            "results": [
                {"lista_palista_id": ["a"], "tiempo": "00:01:00.00"},
                {"lista_palista_id": ["b"], "tiempo": "00:01:10.00"},
            ],
        }
        groups, _ = prepare_performance_groups(
            [{"regata_id": "r1", "fecha": "2025-01-01", "pruebas": [prueba]}]
        )
        fixed = process_performance_group({}, groups[0], k=100.0, field_size_alpha=0.0)
        scaled = process_performance_group({}, groups[0], k=100.0, field_size_alpha=0.5)

        self.assertEqual(fixed.evaluation, scaled.evaluation)
        self.assertNotEqual(fixed.team_base_deltas, scaled.team_base_deltas)


if __name__ == "__main__":
    unittest.main()
