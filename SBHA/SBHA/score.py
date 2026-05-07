import json
import numpy as np
from typing import Dict, List, Tuple


def load_json(file_path: str) -> Dict or List:
    """加载JSON文件（支持字典或列表格式）"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        raise ValueError(f"加载文件 {file_path} 失败：{str(e)}")


def calculate_confidence(current_score: float, history_scores: List[float], eps: float = 1e-8) -> float:
    """
    计算单个模型的置信度（基于历史评分分布）
    公式：c_i(x) = 1 - |s_i(x) - median(S_i)| / (max(S_i) - min(S_i) + eps)
    :param current_score: 当前样本的评分（s_i(x)）
    :param history_scores: 模型的全部历史评分（S_i）
    :param eps: 防止分母为0的极小值
    :return: 置信度（范围[0,1]）
    """
    # 计算历史评分的统计量
    history_median = np.median(history_scores)
    history_max = np.max(history_scores)
    history_min = np.min(history_scores)

    # 计算分子（当前评分与中位数的绝对偏差）
    deviation = abs(current_score - history_median)
    # 计算分母（历史评分范围 + 极小值）
    score_range = history_max - history_min + eps
    # 计算置信度
    confidence = 1 - (deviation / score_range)
    # 确保置信度在[0,1]范围内（极端情况截断）
    return max(0.0, min(1.0, confidence))


def calculate_basic_weight(conf_qwen: float, conf_glm: float, lambda_param: float = 1.0) -> float:
    """
    计算Qwen模型的基础权重（基于置信度）
    公式：a0(x) = exp(λ*c1) / (exp(λ*c1) + exp(λ*c2))
    :param conf_qwen: Qwen模型的置信度（c1）
    :param conf_glm: GLM模型的置信度（c2）
    :param lambda_param: 缩放参数（控制置信度对权重的影响强度）
    :return: Qwen的基础权重（a0(x)）
    """
    exp_qwen = np.exp(lambda_param * conf_qwen)
    exp_glm = np.exp(lambda_param * conf_glm)
    return exp_qwen / (exp_qwen + exp_glm)


def adjust_weight(basic_weight: float, score_qwen: float, score_glm: float, gamma_param: float = 1.0) -> float:
    """
    基于模型分歧调整最终权重
    公式：α(x) = 0.5 + (a0(x) - 0.5) * exp(-γ*d(x))，其中d(x)=|s1 - s2|
    :param basic_weight: Qwen的基础权重（a0(x)）
    :param score_qwen: Qwen当前样本评分（s1）
    :param score_glm: GLM当前样本评分（s2）
    :param gamma_param: 分歧调整强度参数
    :return: Qwen的最终权重（α(x)）
    """
    # 计算模型分歧度（d(x)）
    disagreement = abs(score_qwen - score_glm)
    # 计算分歧调整因子
    adjust_factor = np.exp(-gamma_param * disagreement)
    # 调整最终权重
    final_weight = 0.5 + (basic_weight - 0.5) * adjust_factor
    # 确保权重在[0,1]范围内
    return max(0.0, min(1.0, final_weight))


def calculate_weighted_score(
        qwen_task_score: float,
        glm_task_score: float,
        qwen_history: List[float],
        glm_history: List[float],
        lambda_param: float = 1.0,
        gamma_param: float = 1.0
) -> Tuple[float, Dict]:
    """
    计算单个任务样本的最终加权分数（含中间过程数据）
    :param qwen_task_score: Qwen对当前任务的评分
    :param glm_task_score: GLM对当前任务的评分
    :param qwen_history: Qwen的全部历史评分
    :param glm_history: GLM的全部历史评分
    :param lambda_param: 置信度缩放参数
    :param gamma_param: 分歧调整参数
    :return: 最终加权分数 + 中间结果（置信度、权重、分歧度）
    """
    # 1. 计算两个模型的置信度
    conf_qwen = calculate_confidence(qwen_task_score, qwen_history)
    conf_glm = calculate_confidence(glm_task_score, glm_history)

    # 2. 计算Qwen的基础权重
    basic_weight = calculate_basic_weight(conf_qwen, conf_glm, lambda_param)

    # 3. 计算模型分歧度并调整最终权重
    disagreement = abs(qwen_task_score - glm_task_score)
    final_weight_qwen = adjust_weight(basic_weight, qwen_task_score, glm_task_score, gamma_param)
    final_weight_glm = 1 - final_weight_qwen  # GLM权重 = 1 - Qwen权重

    # 4. 计算最终加权分数
    weighted_score = (final_weight_qwen * qwen_task_score) + (final_weight_glm * glm_task_score)

    # 整理中间结果（用于调试和可解释性）
    intermediate = {
        "qwen_confidence": round(conf_qwen, 4),
        "glm_confidence": round(conf_glm, 4),
        "disagreement": round(disagreement, 4),
        "qwen_basic_weight": round(basic_weight, 4),
        "qwen_final_weight": round(final_weight_qwen, 4),
        "glm_final_weight": round(final_weight_glm, 4)
    }

    return round(weighted_score, 4), intermediate


def process_all_tasks(
        qwen_task_path: str,
        glm_task_path: str,
        qwen_history_path: str,
        glm_history_path: str,
        lambda_param: float = 1.0,
        gamma_param: float = 1.0
) -> List[Dict]:
    """
    批量处理所有任务样本，输出含加权分数的完整结果
    :param qwen_task_path: Qwen任务打分文件路径（含id/task/output/score）
    :param glm_task_path: GLM任务打分文件路径（格式同上）
    :param qwen_history_path: Qwen历史打分文件路径（仅历史得分列表）
    :param glm_history_path: GLM历史打分文件路径（格式同上）
    :param lambda_param: 置信度缩放参数
    :param gamma_param: 分歧调整参数
    :return: 每个任务样本的完整结果（原始信息 + 加权分数 + 中间过程）
    """
    # 1. 加载所有数据
    qwen_tasks = load_json(qwen_task_path)  # 列表，每个元素是{"id":..., "task":..., "output":..., "score":...}
    glm_tasks = load_json(glm_task_path)
    qwen_history = load_json(qwen_history_path)  # 列表，如[85.2, 79.5, ...]
    glm_history = load_json(glm_history_path)

    # 2. 校验任务数据一致性（按id匹配，确保样本数量一致）
    qwen_task_dict = {task["id"]: task for task in qwen_tasks}
    glm_task_dict = {task["id"]: task for task in glm_tasks}
    common_ids = set(qwen_task_dict.keys()) & set(glm_task_dict.keys())
    if len(common_ids) != len(qwen_tasks) or len(common_ids) != len(glm_tasks):
        raise ValueError("Qwen和GLM的任务文件样本数量不匹配或id不唯一")

    # 3. 批量计算每个样本的加权分数
    result = []
    for task_id in sorted(common_ids):
        qwen_task = qwen_task_dict[task_id]
        glm_task = glm_task_dict[task_id]

        # 提取当前任务的评分（确保score为数值类型）
        qwen_score = float(qwen_task["score"])
        glm_score = float(glm_task["score"])

        # 计算加权分数和中间结果
        weighted_score, intermediate = calculate_weighted_score(
            qwen_score, glm_score, qwen_history, glm_history, lambda_param, gamma_param
        )

        # 整理完整结果（保留原始信息 + 新增加权分数和中间过程）
        task_result = {
            "id": task_id,
            "task": qwen_task["task"],  # 假设Qwen和GLM的task描述一致
            "qwen_output": qwen_task["output"],
            "glm_output": glm_task["output"],
            "qwen_raw_score": qwen_score,
            "glm_raw_score": glm_score,
            "weighted_score": weighted_score,
            "intermediate": intermediate  # 中间过程数据（置信度、权重等）
        }
        result.append(task_result)

    return result


def save_result(result: List[Dict], output_path: str) -> None:
    """保存最终结果到JSON文件"""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"结果已保存到：{output_path}")


# ------------------- 示例：调用流程 -------------------
if __name__ == "__main__":
    # 1. 配置文件路径（请替换为你的实际文件路径）
    FILE_PATHS = {
        "qwen_task": "qwen_task_scores.json",  # Qwen任务打分文件
        "glm_task": "glm_task_scores.json",  # GLM任务打分文件
        "qwen_history": "qwen_history_scores.json",  # Qwen历史打分文件
        "glm_history": "glm_history_scores.json",  # GLM历史打分文件
        "output": "weighted_scores_result.json"  # 输出结果文件
    }

    # 2. 配置参数（可根据需求调整）
    LAMBDA_PARAM = 1.0  # 置信度对基础权重的影响强度（默认1.0）
    GAMMA_PARAM = 1.0  # 分歧调整强度（默认1.0，值越大分歧影响越强）

    # 3. 执行批量计算
    try:
        print("开始计算加权分数...")
        weighted_result = process_all_tasks(
            qwen_task_path=FILE_PATHS["qwen_task"],
            glm_task_path=FILE_PATHS["glm_task"],
            qwen_history_path=FILE_PATHS["qwen_history"],
            glm_history_path=FILE_PATHS["glm_history"],
            lambda_param=LAMBDA_PARAM,
            gamma_param=GAMMA_PARAM
        )

        # 4. 保存结果
        save_result(weighted_result, FILE_PATHS["output"])

    except Exception as e:
        print(f"计算过程出错：{str(e)}")