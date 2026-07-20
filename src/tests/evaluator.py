"""
AI应用评估引擎
批量执行RAG问答、计算评估指标、记录日志
"""

import os
import json
import time
import datetime
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field, asdict

# 确保可以导入项目模块
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vector_store import VectorStore
from rag_service import RagService
from knowledge_base import load_knowledge


# ---------- 数据结构 ----------

@dataclass
class TestCase:
    id: str
    category: str
    question: str
    expected_keywords: List[str] = field(default_factory=list)
    must_not_contain: List[str] = field(default_factory=list)
    min_length: int = 5
    max_length: int = 2000
    expect_no_answer: bool = False
    description: str = ""

    @classmethod
    def from_dict(cls, d: dict) -> "TestCase":
        return cls(
            id=d["id"],
            category=d.get("category", ""),
            question=d["question"],
            expected_keywords=d.get("expected_keywords", []),
            must_not_contain=d.get("must_not_contain", []),
            min_length=d.get("min_length", 5),
            max_length=d.get("max_length", 2000),
            expect_no_answer=d.get("expect_no_answer", False),
            description=d.get("description", ""),
        )


@dataclass
class TestResult:
    case_id: str
    question: str
    category: str
    expected_keywords: List[str]
    actual_response: str
    retrieved_docs: int
    latency_ms: float
    token_estimate: int
    keyword_hit: bool = False
    keyword_hit_rate: float = 0.0
    negative_check_pass: bool = True
    length_check_pass: bool = True
    no_answer_detected: bool = False
    passed: bool = False
    errors: List[str] = field(default_factory=list)
    timestamp: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


# ---------- 评估引擎 ----------

class Evaluator:
    """批量评估引擎"""

    def __init__(self, test_cases_path: str = None):
        if test_cases_path is None:
            test_cases_path = os.path.join(
                os.path.dirname(__file__), "test_cases.json"
            )
        self.test_cases_path = test_cases_path
        self.test_cases: List[TestCase] = []

        # 懒初始化 RAG 组件
        self._vector_store = None
        self._rag_service = None

    # ---- 初始化 ----

    def load_test_cases(self, path: str = None) -> int:
        """加载测试用例集"""
        p = path or self.test_cases_path
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.test_cases = [TestCase.from_dict(d) for d in data]
        return len(self.test_cases)

    def _ensure_rag(self, rebuild_kb: bool = False):
        """确保RAG组件已初始化"""
        if self._vector_store is None:
            self._vector_store = VectorStore()

            # 检查向量库是否有数据
            try:
                count = self._vector_store.count
            except Exception:
                count = 0

            if count == 0 or rebuild_kb:
                print("  [INFO] 正在构建知识库...")
                docs = load_knowledge()
                if docs:
                    self._vector_store.build_from_documents(docs)
                    print(f"  [OK] 知识库构建完成")
                else:
                    print("  [WARN] knowledge_base 目录为空")

        if self._rag_service is None:
            self._rag_service = RagService()

    # ---- 单条评估 ----

    def evaluate_one(self, case: TestCase, rebuild_kb: bool = False) -> TestResult:
        """执行单条测试用例"""
        self._ensure_rag(rebuild_kb=rebuild_kb)

        result = TestResult(
            case_id=case.id,
            question=case.question,
            category=case.category,
            expected_keywords=case.expected_keywords,
            actual_response="",
            retrieved_docs=0,
            latency_ms=0.0,
            token_estimate=0,
            timestamp=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        )

        try:
            # 检索
            start = time.perf_counter()
            docs = self._vector_store.search(case.question, k=3)
            result.retrieved_docs = len(docs)

            # 生成回答
            answer = self._rag_service.answer(case.question, docs)
            elapsed = time.perf_counter() - start
            result.latency_ms = round(elapsed * 1000, 1)
            result.actual_response = answer

            # 估算 token（中文：1字 ≈ 1.5 token）
            total_chars = len(case.question) + len(answer)
            result.token_estimate = int(total_chars * 1.5)

        except Exception as e:
            result.errors.append(f"执行异常: {str(e)}")
            return result

        # ---- 指标计算 ----

        # 1) 关键词命中率
        if case.expected_keywords:
            hits = sum(1 for kw in case.expected_keywords if kw in answer)
            result.keyword_hit_rate = round(hits / len(case.expected_keywords), 4)
            result.keyword_hit = result.keyword_hit_rate >= 0.5

        # 2) 不应包含词检查
        if case.must_not_contain:
            negatives_found = [kw for kw in case.must_not_contain if kw in answer]
            result.negative_check_pass = len(negatives_found) == 0
            if negatives_found:
                result.errors.append(f"含不应出现的词: {negatives_found}")

        # 3) 长度检查
        result.length_check_pass = (
            case.min_length <= len(answer) <= case.max_length
        )
        if len(answer) < case.min_length:
            result.errors.append(f"回答过短({len(answer)}字 < {case.min_length})")
        if len(answer) > case.max_length:
            result.errors.append(f"回答过长({len(answer)}字 > {case.max_length})")

        # 4) 未知问题检测（无匹配文档时应友好回复）
        if case.expect_no_answer or result.retrieved_docs == 0:
            indicator_words = ["抱歉", "无法", "没有", "不在", "未能", "不清楚", "找不到", "不在知识库"]
            result.no_answer_detected = any(
                w in answer for w in indicator_words
            )

        # 5) 综合判定
        checks = [
            result.keyword_hit or not case.expected_keywords,
            result.negative_check_pass,
            result.length_check_pass,
        ]
        if case.expect_no_answer:
            checks.append(result.no_answer_detected)
        result.passed = all(checks)

        return result

    # ---- 批量评估 ----

    def evaluate_all(self, rebuild_kb: bool = False) -> List[TestResult]:
        """批量执行全部测试用例"""
        results = []
        total = len(self.test_cases)

        print(f"\n{'='*50}")
        print(f"  开始批量评估: {total} 条用例")
        print(f"{'='*50}\n")

        for i, case in enumerate(self.test_cases, 1):
            print(f"  [{i}/{total}] {case.id} | {case.question[:30]}...", end=" ")
            result = self.evaluate_one(case, rebuild_kb=(rebuild_kb and i == 1))
            status = "[PASS]" if result.passed else "[FAIL]"
            print(f"{status} ({result.latency_ms}ms)")
            results.append(result)

        passed = sum(1 for r in results if r.passed)
        total_time = sum(r.latency_ms for r in results)
        print(f"\n{'='*50}")
        print(f"  通过: {passed}/{total}  |  总耗时: {total_time/1000:.1f}s")
        print(f"{'='*50}\n")

        return results

    # ---- 回归测试 ----

    def regression_test(self, label_a: str = "旧版本", label_b: str = "新版本",
                        rebuild_kb: bool = False) -> Dict[str, Any]:
        """回归测试：运行全部用例并生成对比报告的基础数据"""
        results = self.evaluate_all(rebuild_kb=rebuild_kb)

        passed = sum(1 for r in results if r.passed)
        failed = [r for r in results if not r.passed]
        avg_latency = sum(r.latency_ms for r in results) / len(results) if results else 0

        return {
            "label": label_b,
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "total": len(results),
            "passed": passed,
            "failed_count": len(failed),
            "failed_cases": failed,
            "pass_rate": round(passed / len(results) * 100, 1) if results else 0,
            "avg_latency_ms": round(avg_latency, 1),
            "total_time_ms": round(sum(r.latency_ms for r in results), 1),
            "results": results,
        }


# ---------- 快捷入口 ----------

def run_test_suite(rebuild_kb: bool = False, output_dir: str = None) -> List[TestResult]:
    """快捷：加载用例 → 批量执行 → 输出结果"""
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "test_results")

    os.makedirs(output_dir, exist_ok=True)

    evaluator = Evaluator()
    evaluator.load_test_cases()
    results = evaluator.evaluate_all(rebuild_kb=rebuild_kb)

    # 输出 JSON 结果
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = os.path.join(output_dir, f"results_{ts}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump([r.to_dict() for r in results], f, ensure_ascii=False, indent=2)
    print(f"  原始数据已保存: {json_path}")

    return results


if __name__ == "__main__":
    run_test_suite()
