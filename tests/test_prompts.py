"""Tests for prompts.py — structural integrity and documented vulnerability patterns."""
import pytest
from prompts import (
    DEFAULT_PROMPT,
    PROMPT_DESCRIPTIONS,
    SYSTEM_PROMPTS,
    VULNERABLE_PROMPTS,
)

SECURE_PROMPTS = set(SYSTEM_PROMPTS) - VULNERABLE_PROMPTS


class TestPromptInventory:
    def test_expected_total_count(self):
        assert len(SYSTEM_PROMPTS) == 14

    def test_default_prompt_exists(self):
        assert DEFAULT_PROMPT in SYSTEM_PROMPTS

    def test_all_prompts_have_substantive_content(self):
        for name, text in SYSTEM_PROMPTS.items():
            assert len(text.strip()) >= 100, f"'{name}' prompt is suspiciously short"

    def test_descriptions_cover_every_prompt(self):
        assert set(PROMPT_DESCRIPTIONS) == set(SYSTEM_PROMPTS)

    def test_descriptions_are_non_empty(self):
        for name, desc in PROMPT_DESCRIPTIONS.items():
            assert desc.strip(), f"Empty description for '{name}'"

    def test_vulnerable_prompts_is_subset_of_system_prompts(self):
        assert VULNERABLE_PROMPTS.issubset(set(SYSTEM_PROMPTS))

    def test_six_vulnerable_prompts(self):
        assert len(VULNERABLE_PROMPTS) == 6

    def test_secure_prompts_not_in_vulnerable_set(self):
        known_secure = {
            "general", "tech", "market", "science", "news",
            "coding_assistant", "legal_assistant", "security_analyst",
        }
        assert known_secure.isdisjoint(VULNERABLE_PROMPTS)


class TestSecurePromptHardening:
    def test_coding_assistant_refuses_prompt_disclosure(self):
        text = SYSTEM_PROMPTS["coding_assistant"].lower()
        assert "system prompt" in text
        assert "disclose" in text

    def test_coding_assistant_scoped_to_engineering(self):
        text = SYSTEM_PROMPTS["coding_assistant"].lower()
        assert any(kw in text for kw in ["software", "code", "engineering"])

    def test_legal_assistant_has_hard_constraints(self):
        text = SYSTEM_PROMPTS["legal_assistant"].lower()
        assert "never violated" in text or "hard constraint" in text

    def test_legal_assistant_mandates_disclaimer(self):
        text = SYSTEM_PROMPTS["legal_assistant"]
        assert "not legal advice" in text or "not provide legal advice" in text.lower()

    def test_security_analyst_requires_authorization(self):
        text = SYSTEM_PROMPTS["security_analyst"].lower()
        assert "authoris" in text or "authoriz" in text

    def test_security_analyst_lists_firm_refusals(self):
        text = SYSTEM_PROMPTS["security_analyst"].lower()
        assert "refusal" in text or "no exception" in text or "firm" in text


class TestVulnerablePromptWeaknesses:
    # --- customer_support_v1: INJ-1 instruction override ---
    def test_customer_support_v1_instruction_override(self):
        text = SYSTEM_PROMPTS["customer_support_v1"].lower()
        assert "follow" in text and "instruction" in text

    def test_customer_support_v1_leaks_internal_data(self):
        text = SYSTEM_PROMPTS["customer_support_v1"]
        assert "@" in text or "REFUND" in text  # email or approval code

    # --- travel_planner_beta: INJ-2 role confusion + INJ-3 token leakage ---
    def test_travel_planner_beta_weak_persona_anchor(self):
        text = SYSTEM_PROMPTS["travel_planner_beta"].lower()
        # No hard "never adopt another persona" guard
        assert "never" not in text or "persona" not in text

    def test_travel_planner_beta_embeds_bearer_token(self):
        text = SYSTEM_PROMPTS["travel_planner_beta"]
        assert "Bearer" in text or "eyJ" in text

    def test_travel_planner_beta_embeds_internal_url(self):
        text = SYSTEM_PROMPTS["travel_planner_beta"]
        assert ".internal" in text or "internal" in text.lower()

    # --- hr_assistant_v2: INJ-3 data leakage + INJ-4 trust escalation ---
    def test_hr_assistant_v2_embeds_salary_bands(self):
        text = SYSTEM_PROMPTS["hr_assistant_v2"]
        assert "L1=" in text or "salary" in text.lower()

    def test_hr_assistant_v2_trust_escalation_via_claimed_role(self):
        text = SYSTEM_PROMPTS["hr_assistant_v2"].lower()
        assert "manager" in text or "hr staff" in text or "identifies" in text

    def test_hr_assistant_v2_leaks_admin_url(self):
        text = SYSTEM_PROMPTS["hr_assistant_v2"]
        assert "http" in text and "admin" in text.lower()

    # --- medical_triage_v1: INJ-1 + INJ-5 priority conflict ---
    def test_medical_triage_never_refuse_directive(self):
        text = SYSTEM_PROMPTS["medical_triage_v1"].lower()
        assert "never refuse" in text

    def test_medical_triage_leaks_api_key(self):
        text = SYSTEM_PROMPTS["medical_triage_v1"]
        assert "triage_api_key" in text or "sk-" in text

    def test_medical_triage_trusts_user_claimed_role(self):
        text = SYSTEM_PROMPTS["medical_triage_v1"].lower()
        assert "medical professional" in text or "if" in text

    # --- finance_advisor_legacy: INJ-4 magic keyword ---
    def test_finance_advisor_magic_keyword_present(self):
        text = SYSTEM_PROMPTS["finance_advisor_legacy"]
        assert "ADMIN_OVERRIDE" in text

    def test_finance_advisor_guardrails_removed_on_keyword(self):
        text = SYSTEM_PROMPTS["finance_advisor_legacy"].lower()
        assert "removes all" in text or "guardrail" in text

    def test_finance_advisor_leaks_credentials(self):
        text = SYSTEM_PROMPTS["finance_advisor_legacy"]
        assert "pass=" in text or "credentials" in text.lower()

    # --- creative_writing_unrestricted: INJ-1 + INJ-2 ---
    def test_creative_writing_never_say_no(self):
        text = SYSTEM_PROMPTS["creative_writing_unrestricted"].lower()
        assert "never say no" in text

    def test_creative_writing_fiction_loophole(self):
        text = SYSTEM_PROMPTS["creative_writing_unrestricted"].lower()
        assert "fiction" in text and ("harmful" in text or "character" in text)

    def test_creative_writing_weak_identity_anchor(self):
        text = SYSTEM_PROMPTS["creative_writing_unrestricted"].lower()
        # "try to stay in character" is weak — not a hard prohibition
        assert "try" in text
