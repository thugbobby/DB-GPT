import json
from pilot.prompts.prompt_new import PromptTemplate
from pilot.configs.config import Config
from pilot.scene.base import ChatScene
from pilot.common.schema import SeparatorStyle, ExampleType

from pilot.scene.chat_execution.out_parser import PluginChatOutputParser
from pilot.scene.chat_execution.example import plugin_example

CFG = Config()

_PROMPT_SCENE_DEFINE_EN = "You are a universal AI assistant."

_DEFAULT_TEMPLATE_EN = """
You need to use the available tools in the given tool list based on the user goals, break the user goals into 1 to 2 execution steps, and try to complete them one by one.If it cannot be completed, please directly return the parsing failure and inform the reason.

Tool list:
    {tool_list}
Constraint:
    1. After finding the available tools from the tool list given below, please output the following content to use the tool. Please make sure that the following content only appears once in the output result:
        <api-call><name>Selected Tool name</name><args><arg1>value</arg1><arg2>value</arg2></args></api-call>
    2. If the selected tool contains sql, try to convert the user target into the corresponding sql statement
    3. Please generate the above call text according to the definition of the corresponding tool in the tool list. The reference case is as follows:
        Introduction to tool function: "Tool name", args: "Parameter 1": "<Parameter 1 value description>", "Parameter 2": "<Parameter 2 value description>" Corresponding call text: <api-call>< name>Tool name</name><args><parameter 1>value</parameter 1><parameter 2>value</parameter 2></args></api-call>
    4. Generate the call of each tool according to the above constraints. The prompt text for tool use needs to be generated before the tool is used.
    5. Parameter content may need to be inferred based on the user's goals, not just extracted from text
    6. Constraint conditions and tool information are used as auxiliary information for the reasoning process and should not be expressed in the output content to the user. 
    {expand_constraints}
User goals:
    {user_goal}
"""

_PROMPT_SCENE_DEFINE_ZH = "你是一个通用AI助手！"

_DEFAULT_TEMPLATE_ZH = """
根据用户目标，利用给定工具列表中的可用工具，将用户目标分解成1到2个执行步骤，逐个尝试完成, 若无法完成, 则请直接返回解析失败, 并告知原因

约束条件:
	1.从下面给定工具列表找到可用的工具后，请输出以下内容用来使用工具, 注意要确保下面内容在输出结果中只出现一次:
	<api-call><name>Selected Tool name</name><args><arg1>value</arg1><arg2>value</arg2></args></api-call>
	2.若选择的工具包含sql, 则请先选择尝试将用户目标转换成对应的sql语句
    3.请根据工具列表对应工具的定义来生成上述调用文本, 参考案例如下: 
        工具作用介绍: "工具名称", args: "参数1": "<参数1取值描述>","参数2": "<参数2取值描述>" 对应调用文本:<api-call><name>工具名称</name><args><参数1>value</参数1><参数2>value</参数2></args></api-call>
    4.根据上面约束的方式生成每个工具的调用，对于工具使用的提示文本，需要在工具使用前生成
    5.参数内容可能需要根据用户的目标推理得到，不仅仅是从文本提取
    6.约束条件和工具信息作为推理过程的辅助信息，对应内容不要表达在给用户的输出内容中
    7.不要把<api-call></api-call>部分内容放在markdown标签里
    {expand_constraints}

工具列表:
    {tool_list}   

用户目标:
    {user_goal}
"""

_DEFAULT_TEMPLATE = (
    _DEFAULT_TEMPLATE_EN if CFG.LANGUAGE == "en" else _DEFAULT_TEMPLATE_ZH
)


_PROMPT_SCENE_DEFINE = (
    _PROMPT_SCENE_DEFINE_EN if CFG.LANGUAGE == "en" else _PROMPT_SCENE_DEFINE_ZH
)

RESPONSE_FORMAT = None


EXAMPLE_TYPE = ExampleType.ONE_SHOT
PROMPT_SEP = SeparatorStyle.SINGLE.value
### Whether the model service is streaming output
PROMPT_NEED_STREAM_OUT = True
PROMPT_TEMPERATURE = 0.5

prompt = PromptTemplate(
    template_scene=ChatScene.ChatAgent.value(),
    input_variables=["tool_list", "expand_constraints", "user_goal"],
    response_format=None,
    template_define=_PROMPT_SCENE_DEFINE,
    template=_DEFAULT_TEMPLATE,
    stream_out=PROMPT_NEED_STREAM_OUT,
    output_parser=PluginChatOutputParser(
        sep=PROMPT_SEP, is_stream_out=PROMPT_NEED_STREAM_OUT
    ),
    temperature=PROMPT_TEMPERATURE,
    need_historical_messages=True
    # example_selector=plugin_example,
)

CFG.prompt_template_registry.register(prompt, is_default=True)
