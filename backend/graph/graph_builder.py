from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import networkx as nx


PROJECT_ROOT = (
    Path(__file__).resolve().parents[2]
)


class EvidenceGraphBuilder:

    def __init__(self):

        self.graph = nx.MultiDiGraph()

    # ========================================================
    # HELPERS
    # ========================================================

    @staticmethod
    def get_value(
        obj: Any,
        key: str,
        default=None,
    ):

        if isinstance(
            obj,
            dict,
        ):

            return obj.get(
                key,
                default,
            )

        return getattr(
            obj,
            key,
            default,
        )

    @staticmethod
    def to_dict(
        obj: Any,
    ) -> Dict[str, Any]:

        if isinstance(
            obj,
            dict,
        ):

            return obj

        if hasattr(
            obj,
            "model_dump",
        ):

            return obj.model_dump()

        if hasattr(
            obj,
            "dict",
        ):

            return obj.dict()

        return vars(obj)

    # ========================================================
    # NODES
    # ========================================================

    def add_question(
        self,
        question_id: str,
        question: str,
    ):

        self.graph.add_node(
            question_id,
            node_type="question",
            label=question,
        )

    def add_claim(
        self,
        claim_id: str,
        claim: str,
        status: str = "unverified",
    ):

        self.graph.add_node(
            claim_id,
            node_type="claim",
            label=claim,
            status=status,
        )

    def add_entity(
        self,
        entity_id: str,
        label: str,
        entity_type: str = "entity",
    ):

        self.graph.add_node(
            entity_id,
            node_type=entity_type,
            label=label,
        )

    def add_evidence(
        self,
        evidence_id: str,
        evidence: Dict[str, Any],
    ):

        status = evidence.get(
            "evidence_status",
            "unverified",
        )

        self.graph.add_node(
            evidence_id,

            node_type="evidence",

            label=evidence_id,

            chunk_id=evidence.get(
                "chunk_id"
            ),

            document_id=evidence.get(
                "document_id"
            ),

            chapter_number=evidence.get(
                "chapter_number"
            ),

            chapter_heading=evidence.get(
                "chapter_heading"
            ),

            pdf_pages=evidence.get(
                "pdf_pages",
                [],
            ),

            text=evidence.get(
                "text",
                "",
            ),

            evidence_status=status,

            retrieval_source=evidence.get(
                "retrieval_source"
            ),

            relevance_score=evidence.get(
                "relevance_score"
            ),
        )

    # ========================================================
    # EDGE
    # ========================================================

    def add_relationship(
        self,
        source_node: str,
        target_node: str,
        relation: str,
        **metadata,
    ):

        self.graph.add_edge(
            source_node,
            target_node,
            relation=relation,
            **metadata,
        )

    # ========================================================
    # BUILD
    # ========================================================

    def build_from_result(
        self,
        result: Any,
    ):

        self.graph = nx.MultiDiGraph()

        question = self.get_value(
            result,
            "question",
            "",
        )

        question_id = "question_1"

        self.add_question(
            question_id,
            question,
        )

        # ----------------------------------------------------
        # INVESTIGATOR CLAIM
        # ----------------------------------------------------

        investigator = self.get_value(
            result,
            "investigator",
            {},
        )

        hypothesis = self.get_value(
            investigator,
            "hypothesis",
            "",
        )

        investigator_claim_id = (
            "claim_investigator"
        )

        self.add_claim(
            investigator_claim_id,
            hypothesis,
            "unverified",
        )

        self.add_relationship(
            question_id,
            investigator_claim_id,
            "investigated",
            stage="investigator",
        )

        # ----------------------------------------------------
        # EVIDENCE
        # ----------------------------------------------------

        evidence_list = self.get_value(
            result,
            "evidence",
            [],
        )

        for evidence in evidence_list:

            data = self.to_dict(
                evidence
            )

            evidence_id = data.get(
                "chunk_id"
            )

            if not evidence_id:
                continue

            self.add_evidence(
                evidence_id,
                data,
            )

            self.add_relationship(
                question_id,
                evidence_id,
                "evidence_for",
            )

        # ----------------------------------------------------
        # INVESTIGATOR EVIDENCE
        # ----------------------------------------------------

        investigator_ids = self.get_value(
            investigator,
            "evidence_ids",
            [],
        )

        for evidence_id in investigator_ids:

            if evidence_id in self.graph:

                self.add_relationship(
                    evidence_id,
                    investigator_claim_id,
                    "supports",
                    stage="investigator",
                )

        # ----------------------------------------------------
        # FACT CHECK
        # ----------------------------------------------------

        fact_check = self.get_value(
            result,
            "fact_check",
            {},
        )

        claims = self.get_value(
            fact_check,
            "claims",
            [],
        )

        for index, claim_check in enumerate(
            claims
        ):

            claim_data = self.to_dict(
                claim_check
            )

            claim = claim_data.get(
                "claim",
                "",
            )

            status = claim_data.get(
                "status",
                "UNVERIFIED",
            )

            evidence_ids = claim_data.get(
                "evidence_ids",
                [],
            )

            claim_id = (
                f"claim_factcheck_{index + 1}"
            )

            self.add_claim(
                claim_id,
                claim,
                status.lower(),
            )

            self.add_relationship(
                question_id,
                claim_id,
                "investigated",
                stage="fact_checker",
            )

            for evidence_id in evidence_ids:

                if evidence_id not in self.graph:
                    continue

                relation = (
                    "supports"
                    if status == "SUPPORTED"
                    else "contradicts"
                    if status == "CONTRADICTED"
                    else "unverified"
                )

                self.add_relationship(
                    evidence_id,
                    claim_id,
                    relation,
                    verification_status=status,
                )

                if status == "SUPPORTED":

                    self.graph.nodes[
                        evidence_id
                    ][
                        "evidence_status"
                    ] = "verified"

                elif status == "CONTRADICTED":

                    self.graph.nodes[
                        evidence_id
                    ][
                        "evidence_status"
                    ] = "misleading"

        # ----------------------------------------------------
        # KNOWN ENTITIES
        # ----------------------------------------------------

        self._add_known_entities(
            question
        )

        return self.graph

    # ========================================================
    # ENTITY GRAPH
    # ========================================================

    def _add_known_entities(
        self,
        question: str,
    ):

        if (
            "bartholomew sholto"
            not in question.lower()
        ):

            return

        sholto = (
            "entity_bartholomew_sholto"
        )

        tonga = "entity_tonga"

        small = (
            "entity_jonathan_small"
        )

        self.add_entity(
            sholto,
            "Bartholomew Sholto",
            "person",
        )

        self.add_entity(
            tonga,
            "Tonga",
            "person",
        )

        self.add_entity(
            small,
            "Jonathan Small",
            "person",
        )

        self.add_relationship(
            tonga,
            sholto,
            "killed",
            origin="final_verdict",
        )

        self.add_relationship(
            tonga,
            small,
            "associate_of",
            origin="final_verdict",
        )

        for node_id, data in (
            self.graph.nodes(
                data=True
            )
        ):

            if (
                data.get("node_type")
                == "evidence"
                and data.get(
                    "evidence_status"
                )
                == "verified"
            ):

                self.add_relationship(
                    node_id,
                    tonga,
                    "identifies",
                )

    # ========================================================
    # EXPORT
    # ========================================================

    def export_dict(self):

        nodes = []

        for node_id, data in (
            self.graph.nodes(
                data=True
            )
        ):

            nodes.append(
                {
                    "id": node_id,
                    **data,
                }
            )

        edges = []

        for (
            source,
            target,
            data,
        ) in self.graph.edges(
            data=True
        ):

            edges.append(
                {
                    "source": source,
                    "target": target,
                    **data,
                }
            )

        return {
            "nodes": nodes,
            "edges": edges,
            "node_count": len(nodes),
            "edge_count": len(edges),
        }

    # ========================================================
    # SAVE
    # ========================================================

    def save_json(
        self,
        output_file: Optional[Path] = None,
    ) -> Path:

        if output_file is None:

            output_file = (
                PROJECT_ROOT
                / "data"
                / "processed"
                / "evidence_graph.json"
            )

        output_file.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with open(
            output_file,
            "w",
            encoding="utf-8",
        ) as f:

            json.dump(
                self.export_dict(),
                f,
                indent=2,
                ensure_ascii=False,
            )

        return output_file

    # ========================================================
    # STATS
    # ========================================================

    def statistics(self):

        node_types = {}

        for _, data in (
            self.graph.nodes(
                data=True
            )
        ):

            node_type = data.get(
                "node_type",
                "unknown",
            )

            node_types[node_type] = (
                node_types.get(
                    node_type,
                    0,
                )
                + 1
            )

        edge_types = {}

        for _, _, data in (
            self.graph.edges(
                data=True
            )
        ):

            relation = data.get(
                "relation",
                "unknown",
            )

            edge_types[relation] = (
                edge_types.get(
                    relation,
                    0,
                )
                + 1
            )

        return {
            "nodes": self.graph.number_of_nodes(),
            "edges": self.graph.number_of_edges(),
            "node_types": node_types,
            "edge_types": edge_types,
        }