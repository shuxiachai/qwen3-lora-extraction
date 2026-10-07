"""生成完全虚构、组级隔离的投诉抽取练习数据。"""
from __future__ import annotations
import json
import random
import re
from pathlib import Path

SEED = 42
FIELDS = ("name", "address", "email", "question")
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
NAMES = ["林晓雨", "周明远", "沈星河", "贺知秋", "唐云帆", "苏晚晴", "顾南枝", "陆青禾", "程一诺", "许清澜", "白子墨", "姜海棠", "叶闻溪", "宋知遥", "方亦安", "罗嘉言", "魏若岚", "邵景初", "温书瑶", "谢临川", "黎静宜", "乔远山", "韩雨棠", "钟以宁", "杜星野"]
ISSUES = ["上周购买的净水壶漏水，客服三天没有回复。", "新装的星网宽带连续掉线，报修后仍未处理。", "预付卡扣费两次，账单和实际消费对不上。", "维修人员约好时间却没有上门，也未通知。", "订单显示已签收，但我从未收到包裹。", "会员取消后仍持续扣款，请核查退款。", "电饭锅首次使用就跳闸，售后拒绝换货。", "预约的体检项目被取消，前台无法说明原因。", "快递柜取件码失效，包裹被退回。", "在线课程无法播放，客服只回复模板消息。", "水费账单金额异常，比平时高出很多。", "停车场自动扣费失败却显示欠费。", "手机屏幕保修期内出现亮线，门店拒收。", "航班改签后座位消失，平台要求再次付费。", "洗衣机安装后漏水，工单一直显示处理中。", "药店配送少了一盒商品，申请补发无结果。", "物业门禁反复失灵，夜间无法进入小区。", "燃气检修预约被取消，未给新的时间。", "共享单车已还车仍在计费，申诉未处理。", "餐厅外卖严重缺少餐品，商家拒绝退款。", "保险咨询电话被反复挂断，无法办理变更。", "图书馆借阅记录错误，系统显示逾期。", "健身房停业后仍扣月费，联系方式无人接听。", "家电延保合同未生效却收取服务费。", "小区照明故障两周未修，晚归存在安全隐患。"]
LAYOUTS = [
"市民热线记录：{n}{a}{e}反映：{q}", "邮件主题【服务申诉】。{n}{e}{a}正文：{q}", "受理单编号 Z-03。{q}{n}{a}{e}", "我在留言簿写道：{q}{n}{e}{a}", "社区转办件\n{n}\n{a}\n{e}\n问题如下：{q}", "客服工单摘要——{n}{q}{a}{e}", "请核实以下情况：{q}署名信息：{n}{a}{e}", "《居民意见表》{n}住址栏：{a}联络栏：{e}意见：{q}", "录音转写：先说明{q}来电人为{n}，地址是{a}，电邮为{e}", "售后登记：{a}{n}{e}故障描述：{q}", "[便民平台] {q}；身份资料：{n}{a}{e}", "现场笔录：{n}陈述“{q}”联系处：{a}{e}", "投诉卡正面\n{n}\n{q}\n投诉卡背面\n{a}{e}", "我的诉求是：{q}本人资料：{n}{e}{a}", "转交部门的事项：{q}\n申报者：{n}\n居所：{a}\n邮件：{e}", "值班日志记载，{n}称：{q}核对地址{a}；回邮{e}", "在线表单 / 投诉正文：{q} / {n}{a}{e}", "请阅读附件说明：{n}联系地点：{a}邮箱：{e}具体问题：{q}", "窗口受理语音：{q}以上由{n}提交，{a}{e}", "编号十九的申请：{a}{e}{n}申请缘由：{q}", "居民信箱来件\n缘由：{q}\n{n}{a}{e}", "事项速记：{n}{q}{e}{a}", "服务质量反馈：{q}；填表人资料为{n}{a}{e}", "告知书附页：{a}{n}{e}请求处理：{q}", "问题清单第一项：{q}登记信息依次为{n}{a}{e}"]

# 槽位包含自己的字段说明；外层模板不另行指认来电人或提交人为投诉人。
LAYOUTS[8] = "录音转写：先说明{q}随后核对身份。{n}最后核对联系方式。{a}{e}转写结束。"
LAYOUTS[11] = "现场笔录：所记载的事项为“{q}”以下为身份核对结果：{n}联系处登记如下：{a}{e}"
LAYOUTS[15] = "值班日志记载，待处理事项是：{q}核对身份的结果是{n}地址记录为{a}回邮记录为{e}"
LAYOUTS[18] = "窗口受理语音：{q}接下来读回登记信息。{n}{a}{e}以上资料已随事项入档。"
SUFFIXES = ("", "补充：此前已通过常规渠道反映。", "登记时间为周三上午。", "请按原联系信息回复。")
PRAISE = "这是一条满意度表扬，感谢工作人员耐心服务。"
NOTICE = "会议定在云岚市明日举行，附件为空。"


def _output_for(mode, name, address, email, issue):
    output = {"name": name, "address": address, "email": email, "question": issue}
    if mode == 1:
        output["email"] = None
    elif mode == 2:
        output["address"] = None
    elif mode in (3, 6):
        output.update(name=None, address=None, email=None)
    elif mode == 4:
        output["question"] = None
    elif mode == 5:
        output.update(address=None, email=None)
    elif mode == 7:
        output = {field: None for field in FIELDS}
    return output


def _render_normal(group, variant, output, mode):
    n = f"投诉人：{output['name']}。" if output["name"] is not None else "投诉人姓名未登记。"
    a = f"投诉人地址：{output['address']}。" if output["address"] is not None else "投诉人地址未登记。"
    e = f"投诉人邮箱：{output['email']}。" if output["email"] is not None else "投诉人邮箱未登记。"
    if mode == 3:
        n = f"转述人：{NAMES[(group + 7) % 25]}（并非投诉人），实际投诉人姓名未登记。"
    elif mode == 6:
        n = f"涉及人员：{NAMES[(group + 7) % 25]}、{NAMES[(group + 9) % 25]}，未说明谁是投诉人。"
    elif mode == 7:
        n = f"记录员：{NAMES[(group + 3) % 25]}（并非投诉人），本件没有投诉人。"
    if mode == 5:
        a = f"承办点地址：云岚市虚构路{group + 1}号服务站。投诉人地址未登记。"
        e = f"承办点邮箱：service{group + 1:02d}@example.com。投诉人邮箱未登记。"
    q = output["question"] if output["question"] is not None else (NOTICE if mode == 7 else PRAISE)
    return LAYOUTS[group].format(n=n, a=a, e=e, q=q) + SUFFIXES[variant]


def template_signature(text):
    """归一化实体、事项和已知可选槽位，保留外围模板以检查模板跨组。"""
    for suffix in SUFFIXES[1:]:
        if text.endswith(suffix):
            text = text[:-len(suffix)]
            break
    for value in sorted(NAMES + ISSUES + [PRAISE, NOTICE], key=len, reverse=True):
        text = text.replace(value, "<value>")
    text = re.sub(r"[A-Za-z0-9_]+@example\.com", "<email>", text)
    text = re.sub(r"云岚市星桥区幻月街\d+号雾港苑\d+室", "<address>", text)
    text = re.sub(r"承办点地址：云岚市虚构路\d+号服务站。", "", text)
    text = re.sub(r"承办点邮箱：<email>。", "", text)
    name_forms = (
        "投诉人：<value>。", "投诉人姓名未登记。",
        "转述人：<value>（并非投诉人），实际投诉人姓名未登记。",
        "涉及人员：<value>、<value>，未说明谁是投诉人。",
        "记录员：<value>（并非投诉人），本件没有投诉人。",
    )
    for value in sorted(name_forms, key=len, reverse=True):
        text = text.replace(value, "<name-slot>")
    for field, placeholder in (("地址", "<address>"), ("邮箱", "<email>")):
        for value in (f"投诉人{field}：{placeholder}。", f"投诉人{field}未登记。"):
            text = text.replace(value, f"<{field}-slot>")
    return re.sub(r"\s+", "", text)


def make_record(group, variant):
    name = NAMES[group]
    address = f"云岚市星桥区幻月街{group + 1}号雾港苑{variant + 2}室"
    email = f"complain{group + 1:02d}_{variant + 1}@example.com"
    issue = ISSUES[group]
    mode = (group + variant) % 8
    output = _output_for(mode, name, address, email, issue)
    text = _render_normal(group, variant, output, mode)
    return {"id": f"c{group + 1:02d}-{variant + 1}", "group_id": f"source-{group + 1:02d}", "input": text, "output": output}


def build_records():
    return [make_record(group, variant) for group in range(25) for variant in range(4)]


def split_records(records):
    groups = sorted({record["group_id"] for record in records})
    random.Random(SEED).shuffle(groups)
    assignment = {group: "train" if index < 15 else "val" if index < 20 else "test" for index, group in enumerate(groups)}
    return {split: [record for record in records if assignment[record["group_id"]] == split] for split in ("train", "val", "test")}


def main():
    DATA.mkdir(exist_ok=True)
    splits = split_records(build_records())
    for split, records in splits.items():
        (DATA / f"{split}.jsonl").write_text("\n".join(json.dumps(item, ensure_ascii=False, separators=(",", ":")) for item in records) + "\n", encoding="utf-8")
    manifest = {
        "seed": SEED,
        "synthetic": True,
        "fields": list(FIELDS),
        "split_counts": {key: len(value) for key, value in splits.items()},
        "group_counts": {key: len({row["group_id"] for row in value}) for key, value in splits.items()},
        "grouping": "25 source layouts; all four variants retain their group's layout, including missing and ambiguous identity cases; normalized templates are disjoint across groups",
        "test_policy": "quality checks only during preparation; never use test data for prompt selection",
    }
    (DATA / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
