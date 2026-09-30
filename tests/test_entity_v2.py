from __future__ import annotations

import unittest

from mint_synth.entity import entity_block_key, resolution_priority, rule_resolution, validate_llm_clusters


class EntityV2Tests(unittest.TestCase):
    def test_legal_suffixes_share_a_block(self) -> None:
        self.assertEqual(entity_block_key("Example Corporation"), entity_block_key("Example Corp."))
        self.assertEqual(entity_block_key("示例股份有限公司"), entity_block_key("示例有限公司"))

    def test_identifier_precedes_name(self) -> None:
        result = rule_resolution(
            {"entity_surface": "Example", "source_metadata": {"lei": "ABC-123"}},
            aliases={},
            identifier_fields=["lei"],
        )
        self.assertEqual(result["resolution_method"], "identifier:lei")

    def test_identifier_precedes_configured_alias(self) -> None:
        result = rule_resolution(
            {"entity_surface": "Example", "source_metadata": {"lei": "ABC-123"}},
            aliases={
                "example": {
                    "canonical_entity_id": "entity_alias",
                    "canonical_entity_name": "Alias Target",
                }
            },
            identifier_fields=["lei"],
        )
        self.assertEqual(result["resolution_method"], "identifier:lei")
        self.assertNotEqual(result["canonical_entity_id"], "entity_alias")

    def test_identifier_priority_is_preserved_when_surface_facts_are_merged(self) -> None:
        methods = ["alias", "identifier:lei", "normalized_name"]
        self.assertEqual(max(methods, key=resolution_priority), "identifier:lei")

    def test_llm_clusters_must_be_total(self) -> None:
        with self.assertRaisesRegex(ValueError, "cover"):
            validate_llm_clusters(
                {"clusters": [{"canonical_name": "A", "members": ["A"]}]},
                ["A", "A Inc"],
            )


if __name__ == "__main__":
    unittest.main()
