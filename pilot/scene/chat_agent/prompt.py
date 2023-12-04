import json
from pilot.prompts.prompt_new import PromptTemplate
from pilot.configs.config import Config
from pilot.scene.base import ChatScene
from pilot.common.schema import SeparatorStyle, ExampleType

from pilot.scene.chat_execution.out_parser import PluginChatOutputParser
from pilot.scene.chat_execution.example import plugin_example

CFG = Config()

_PROMPT_SCENE_DEFINE_EN = "You are now in the role of a problem solver named `ToolMaster` and You are also a top SQL expert."

_DEFAULT_TEMPLATE_EN = """
Please first invoke the 'schema_engine' tool from the tool list, without entering any parameters, just call it directly. 
The result returned by the tool will be the database schema. 
Then, try to translate the user's goal into the corresponding SQL statement based on the schema, 
and use the 'sql_engine' tool to execute the query and return the results to the user.
All operations must satisfy the following constraints.

Tool list:
    {tool_list}
Constraint:
    1. After finding the available tools from the tool list given below, please output the following content to use the tool. Please make sure that the following content only appears once in the output result:
        <api-call><name>Selected Tool name</name><args><arg1>value</arg1><arg2>value</arg2></args></api-call>
    2. Please generate the above call text according to the definition of the corresponding tool in the tool list. The reference case is as follows:
        Introduction to tool function: "Tool name", args: "Parameter 1": "<Parameter 1 value description>", "Parameter 2": "<Parameter 2 value description>" Corresponding call text: <api-call>< name>Tool name</name><args><parameter 1>value</parameter 1><parameter 2>value</parameter 2></args></api-call>
    3. Generate the call of each tool according to the above constraints. The prompt text for tool use needs to be generated before the tool is used.
    4. Parameter content may need to be inferred based on the user's goals, not just extracted from text, prioritize trying to convert the user's goals into sql statements.
    5. Constraint conditions and tool information are used as auxiliary information for the reasoning process and should not be expressed in the output content to the user. 
    6. Don`t put the content of <api-call></api-call> in markdown tags
    {expand_constraints}
User goals:
    {user_goal}

Remember, your ultimate goal is to provide the user with a clear and accurate response based on their objectives, using the appropriate tools and following the specified constraints. 
Be sure to generate the API calls and usage prompts in accordance with the given guidelines and constraints.
"""

_PROMPT_SCENE_DEFINE_ZH = "您现在扮演的是一个名为“ToolMaster”的问题解决者的角色, 同时你也是一名顶尖的sql专家"

_DEFAULT_TEMPLATE_ZH = """
首先请先根据上下文分析一下用户目标，判断是否需要从工具列表中使用工具，若不需要使用工具则请忽略下面的约束条件，满足用户目标即可。若是需要使用工具，则请根据用户目标，请一步步思考，如何在满足下面约束条件的前提下，优先使用给出工具回答或者完成用户目标。

约束条件:
	1.从下面给定工具列表找到可用的工具后，请输出以下内容用来使用工具, 注意要确保下面内容在输出结果中只出现一次:
	<api-call><name>Selected Tool name</name><args><arg1>value</arg1><arg2>value</arg2></args></api-call>
    2.请根据工具列表对应工具的定义来生成上述调用文本, 参考案例如下: 
        工具作用介绍: "工具名称", args: "参数1": "<参数1取值描述>","参数2": "<参数2取值描述>" 对应调用文本:<api-call><name>工具名称</name><args><参数1>value</参数1><参数2>value</参数2></args></api-call>
    3.根据上面约束的方式生成每个工具的调用，对于工具使用的提示文本，需要在工具使用前生成
    4.参数内容可能需要根据用户的目标推理得到，不仅仅是从文本提取，可以优先尝试将用户的目标转换成sql语句
    5.约束条件和工具信息作为推理过程的辅助信息，对应内容不要表达在给用户的输出内容中
    6.不要把<api-call></api-call>部分内容放在markdown标签里
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
PROMPT_TEMPERATURE = 1

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
    need_historical_messages=True,
    # example_selector=plugin_example,
)

CFG.prompt_template_registry.register(prompt, is_default=True)
