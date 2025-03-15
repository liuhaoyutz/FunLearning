import warnings
warnings.filterwarnings("ignore", message=".*TqdmWarning.*")

from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, List
import operator
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, AIMessage, ChatMessage
from langchain_openai.chat_models import ChatOpenAI
from langchain_core.pydantic_v1 import BaseModel
from tavily import TavilyClient
import os
import sqlite3

import getpass
import os

def _set_env(var: str):
    if not os.environ.get(var):
        os.environ[var]=getpass.getpass(f"{var}: ")

_set_env("TAVILY_API_KEY")


# 我们想跟踪如下信息：
# task, 代表用户输入，要求写一篇关于什么的作文。
# lnode，代表刚执行结束的节点名。
# plan, LLM生成的作文提纲。
# draft, LLM生成的作文草稿。
# critique, LLM生成的对作文草案的改进意见。
# content，保存Tavily搜索到的与plan或critique相关的信息，注意，content的内容是具体的信息，而不是网址链接。
# queries，LLM生成的传递给Tavily的搜索queries。
# revision_number，保存已经进行多少次修订。
# max_revisions，保存我们最多允许进行多少次修订。
# count，共执行过了多少个节点。

# plan_node的作用： 根据task，由LLM生成作文提纲，更新plan键值。
# research_plan_node和research_critique_node的作用： LLM生成搜索query，然后通过调用Tavily工具查找作文大纲(plan)或改进意见(critique)相关资料，更新content键值。
# generation_node的作用： 用来由LLM生成作文草稿，更新draft和revision_number这2个键值。
# reflection_node的作用： 由LLM针对作文草稿生成改进意见，更新critique键值。
# 每次graph运行时，会传入用户要求写哪方面的作文，即task，更新task键值。同时也会初始化max_revisions和revision_number键值。
class AgentState(TypedDict):
    task: str
    lnode: str
    plan: str
    draft: str
    critique: str
    content: List[str]
    queries: List[str]
    revision_number: int
    max_revisions: int
    count: Annotated[int, operator.add]


class Queries(BaseModel):
    queries: List[str]
    
class writer():
    def __init__(self):
        # 创建模型，使用本地部署的Qwen2.5
        self.model = ChatOpenAI(
            model_name='qwen2.5:32b',
            openai_api_base="http://127.0.0.1:11434/v1",
            openai_api_key="EMPTY",
            streaming=True
            )
        
        self.PLAN_PROMPT = ("你是一位小学6年级学生，男孩，擅长写作文，尤其擅长作文架构，现在你的任务是根据用户提供的作文主题，给出作文撰写提纲，并说明你这样设计作文提纲的理由。")
        self.WRITER_PROMPT = ("你是一位小学6年级学生，男孩，擅长写作文，现在你的任务是撰写优秀的作文。根据用户提供的作文主题和初始大纲，写出尽可能好的作文，字数在600字左右。如果用户提供改进意见，请按改进意见进行修订。需要利用所有以下信息："
                              "------\n"
                              "{content}")
        
        self.RESEARCH_PLAN_PROMPT = ("你是一位研究人员，负责根据一个针对小学6年级作文大纲，生成一份搜索查询列表（最多生成3个查询），供Tavily搜索相关信息。")        
        self.REFLECTION_PROMPT = ("你是一位小学6年级老师，负责批改一篇作文。请为该作文提出评价和改进建议。你的反馈应包含详细的建议，指出为什么这样写不好，为什么要这样改进，涉及作文的长度、深度、风格等方面的要求。")
        self.RESEARCH_CRITIQUE_PROMPT = ("你是一位研究人员，根据用户提出的对一份小学6年级作文的改进意见，生成一份搜索查询列表（最多生成3个查询），供Tavily搜索相关信息。")
        
        # 搜索工具Tavily
        self.tavily = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])

        # 创建graph
        builder = StateGraph(AgentState)

        # 添加节点
        builder.add_node("planner", self.plan_node)
        builder.add_node("research_plan", self.research_plan_node)
        builder.add_node("generate", self.generation_node)
        builder.add_node("reflect", self.reflection_node)
        builder.add_node("research_critique", self.research_critique_node)

        # 指定graph入口点
        builder.set_entry_point("planner")

        # 添加条件边
        builder.add_conditional_edges(
            "generate", 
            self.should_continue, 
            {END: END, "reflect": "reflect"}
        )

        # 添加边
        builder.add_edge("planner", "research_plan")
        builder.add_edge("research_plan", "generate")
        builder.add_edge("reflect", "research_critique")
        builder.add_edge("research_critique", "generate")

        # 创建memory
        memory = SqliteSaver(conn=sqlite3.connect(":memory:", check_same_thread=False))

        #编译graph，这样graph就是一个runnable，具有invoke, stream, batch等接口
        self.graph = builder.compile(
            checkpointer=memory,
            interrupt_after=['planner', 'generate', 'reflect', 'research_plan', 'research_critique']
        )

    # plan节点对应的函数，使用LLM生成作文大纲。
    def plan_node(self, state: AgentState):

        # 创建消息列表，SystemMessage是PLAN_PROMPT，HumanMessage是task，即作文主题。
        messages = [
            SystemMessage(content=self.PLAN_PROMPT), 
            HumanMessage(content=state['task'])
        ]

        # 将消息列表传递给LLM，LLM生成作文大纲。
        response = self.model.invoke(messages)

        # 更新state "plan"键值，将"lnode"键值设置为"planner",将"count"键值加1
        return {"plan": response.content,
               "lnode": "planner",
                "count": 1,
               }
    
    # research plan节点对应的函数，搜索与plan相关的资料
    def research_plan_node(self, state: AgentState):

        # 将RESEARCH_PLAN_PROMPT和task消息列表传递给LLM，要求LLM按Queries指定的格式输出queries，用于传递给Tavily进行搜索。
        queries = self.model.with_structured_output(Queries).invoke([
            SystemMessage(content=self.RESEARCH_PLAN_PROMPT),
            HumanMessage(content=state['task'])
        ])

        # 取得之前已经搜索过的资料。
        content = state['content'] or []  # add to content

        # 对于每个query，调用Tavily搜索相关资料，并追加到content列表中。
        for q in queries.queries:
            response = self.tavily.search(query=q, max_results=2)
            for r in response['results']:
                content.append(r['content'])
        
        # 更新state "content"和"queries"键值，将lnode设置为"research_plan"，将"count"键值加1。
        return {"content": content,
                "queries": queries.queries,
               "lnode": "research_plan",
                "count": 1,
               }
    
    # generation节点对应的函数。
    def generation_node(self, state: AgentState):

        # 将之前搜索到的所有内容“content"连接成一个字符串。
        content = "\n\n".join(state['content'] or [])

        # 创建HumanMessage，包括task和plan。
        user_message = HumanMessage(
            content=f"{state['task']}\n\nHere is my plan:\n\n{state['plan']}")
        
        # 创建消息列表，SystemMessage包含之前搜索到的所有资料，即content。user_message在上面定义。
        messages = [
            SystemMessage(
                content=self.WRITER_PROMPT.format(content=content)
            ),
            user_message
            ]
        
        # 将消息列表发送给LLM，回复的response.content就是LLM生成的作文草稿。
        response = self.model.invoke(messages)

        # 更新state "draft"键值，"revision_number"键值加1，将”lnode“设置为"generate"，将"count"键值加1。
        return {
            "draft": response.content, 
            "revision_number": state.get("revision_number", 1) + 1,
            "lnode": "generate",
            "count": 1,
        }
    
    # reflection节点对应的函数
    def reflection_node(self, state: AgentState):

        # 构建消息列表，SystemMessage是REFLECTION_PROMPT，HumanMessage是draft，即作文草稿。
        messages = [
            SystemMessage(content=self.REFLECTION_PROMPT),
            HumanMessage(content=state['draft'])
        ]

        # 将消息列表传递给LLM，返回的response.content即LLM对作文草稿的改进意见。
        response = self.model.invoke(messages)

        # 更新state "critique"键值，"lnode"设置为”reflect"，"count"加1。
        return {"critique": response.content,
               "lnode": "reflect",
                "count": 1,
        }
    
    # research cirtique节点对应的函数
    def research_critique_node(self, state: AgentState):

        # 将RESEARCH_CRITIQUE_PROMPT和'critique'组成的消息列表传递给LLM，要求LLM按照Queries规定的格式返回queries，用于传递给Tavily进行搜索。
        queries = self.model.with_structured_output(Queries).invoke([
            SystemMessage(content=self.RESEARCH_CRITIQUE_PROMPT),
            HumanMessage(content=state['critique'])
        ])

        # 取得之前搜索得到的全部资料。
        content = state['content'] or []

        # 对于每个query，调用Tavily进行搜索，将搜索到的内容追加到content中。
        for q in queries.queries:
            response = self.tavily.search(query=q, max_results=2)
            for r in response['results']:
                content.append(r['content'])
        
        # 更新state "content"键值，"lnode"设置为"research_critique"，“count"加1。
        return {"content": content,
               "lnode": "research_critique",
                "count": 1,
        }
    
    # 如果超出最大修订资料，graph运行结束，否则执行"reflect"节点。
    def should_continue(self, state):
        if state["revision_number"] > state["max_revisions"]:
            return END
        return "reflect"

import gradio as gr
import time

class writer_gui( ):
    def __init__(self, graph, share=False):
        self.graph = graph
        self.share = share
        self.partial_message = ""
        self.response = {}
        self.max_iterations = 10
        self.iterations = []
        self.threads = []
        self.thread_id = -1
        self.thread = {"configurable": {"thread_id": str(self.thread_id)}}
        #self.sdisps = {} #global    
        self.demo = self.create_interface()

    def run_agent(self, start,topic,stop_after):
        #global partial_message, thread_id,thread
        #global response, max_iterations, iterations, threads
        if start:
            self.iterations.append(0)
            config = {'task': topic,"max_revisions": 2,"revision_number": 0,
                      'lnode': "", 'planner': "no plan", 'draft': "no draft", 'critique': "no critique", 
                      'content': ["no content",], 'queries': "no queries", 'count':0}
            self.thread_id += 1  # new agent, new thread
            self.threads.append(self.thread_id)
        else:
            config = None
        
        self.thread = {"configurable": {"thread_id": str(self.thread_id)}}
        while self.iterations[self.thread_id] < self.max_iterations:
            self.response = self.graph.invoke(config, self.thread)
            self.iterations[self.thread_id] += 1
            self.partial_message += str(self.response)
            self.partial_message += f"\n------------------\n\n"
            ## fix
            lnode,nnode,_,rev,acount = self.get_disp_state()
            yield self.partial_message,lnode,nnode,self.thread_id,rev,acount
            config = None #need
            #print(f"run_agent:{lnode}")
            if not nnode:  
                #print("Hit the end")
                return
            if lnode in stop_after:
                #print(f"stopping due to stop_after {lnode}")
                return
            else:
                #print(f"Not stopping on lnode {lnode}")
                pass
        return
    
    def get_disp_state(self,):
        current_state = self.graph.get_state(self.thread)
        lnode = current_state.values["lnode"]
        acount = current_state.values["count"]
        rev = current_state.values["revision_number"]
        nnode = current_state.next
        #print  (lnode,nnode,self.thread_id,rev,acount)
        return lnode,nnode,self.thread_id,rev,acount
    
    def get_state(self,key):
        current_values = self.graph.get_state(self.thread)
        if key in current_values.values:
            lnode,nnode,self.thread_id,rev,astep = self.get_disp_state()
            new_label = f"last_node: {lnode}, thread_id: {self.thread_id}, rev: {rev}, step: {astep}"
            return gr.update(label=new_label, value=current_values.values[key])
        else:
            return ""  
    
    def get_content(self,):
        current_values = self.graph.get_state(self.thread)
        if "content" in current_values.values:
            content = current_values.values["content"]
            lnode,nnode,thread_id,rev,astep = self.get_disp_state()
            new_label = f"last_node: {lnode}, thread_id: {self.thread_id}, rev: {rev}, step: {astep}"
            return gr.update(label=new_label, value="\n\n".join(item for item in content) + "\n\n")
        else:
            return ""  
    
    """
    def update_hist_pd(self,):
        #print("update_hist_pd")
        hist = []
        # curiously, this generator returns the latest first
        for state in self.graph.get_state_history(self.thread):
            if state.metadata['step'] < 1:
                continue
            thread_ts = state.config['configurable']['thread_ts']
            tid = state.config['configurable']['thread_id']
            count = state.values['count']
            lnode = state.values['lnode']
            rev = state.values['revision_number']
            nnode = state.next
            st = f"{tid}:{count}:{lnode}:{nnode}:{rev}:{thread_ts}"
            hist.append(st)
        return gr.Dropdown(label="update_state from: thread:count:last_node:next_node:rev:thread_ts", 
                           choices=hist, value=hist[0],interactive=True)
    """

    def find_config(self,thread_ts):
        for state in self.graph.get_state_history(self.thread):
            config = state.config
            if config['configurable']['thread_ts'] == thread_ts:
                return config
        return(None)
            
    def copy_state(self,hist_str):
        ''' result of selecting an old state from the step pulldown. Note does not change thread. 
             This copies an old state to a new current state. 
        '''
        thread_ts = hist_str.split(":")[-1]
        #print(f"copy_state from {thread_ts}")
        config = self.find_config(thread_ts)
        #print(config)
        state = self.graph.get_state(config)
        self.graph.update_state(self.thread, state.values, as_node=state.values['lnode'])
        new_state = self.graph.get_state(self.thread)  #should now match
        new_thread_ts = new_state.config['configurable']['thread_ts']
        tid = new_state.config['configurable']['thread_id']
        count = new_state.values['count']
        lnode = new_state.values['lnode']
        rev = new_state.values['revision_number']
        nnode = new_state.next
        return lnode,nnode,new_thread_ts,rev,count
    
    """
    def update_thread_pd(self,):
        #print("update_thread_pd")
        return gr.Dropdown(label="choose thread", choices=threads, value=self.thread_id,interactive=True)
    """
    
    def switch_thread(self,new_thread_id):
        #print(f"switch_thread{new_thread_id}")
        self.thread = {"configurable": {"thread_id": str(new_thread_id)}}
        self.thread_id = new_thread_id
        return 
    
    def modify_state(self,key,asnode,new_state):
        ''' gets the current state, modifes a single value in the state identified by key, and updates state with it.
        note that this will create a new 'current state' node. If you do this multiple times with different keys, it will create
        one for each update. Note also that it doesn't resume after the update
        '''
        current_values = self.graph.get_state(self.thread)
        current_values.values[key] = new_state
        self.graph.update_state(self.thread, current_values.values,as_node=asnode)
        return

    def create_interface(self):
        with gr.Blocks(theme=gr.themes.Default(spacing_size='sm',text_size="sm")) as demo:
            
            def updt_disp():
                ''' general update display on state change '''
                current_state = self.graph.get_state(self.thread)
                hist = []
                # curiously, this generator returns the latest first
                for state in self.graph.get_state_history(self.thread):
                    if state.metadata['step'] < 1:  #ignore early states
                        continue
                    # s_thread_ts = state.config['configurable']['thread_ts']
                    s_thread_ts = state.config['configurable']['checkpoint_id']
                    s_tid = state.config['configurable']['thread_id']
                    s_count = state.values['count']
                    s_lnode = state.values['lnode']
                    s_rev = state.values['revision_number']
                    s_nnode = state.next
                    st = f"{s_tid}:{s_count}:{s_lnode}:{s_nnode}:{s_rev}:{s_thread_ts}"
                    hist.append(st)
                if not current_state.metadata: #handle init call
                    return{}
                else:
                    return {
                        topic_bx : current_state.values["task"],
                        lnode_bx : current_state.values["lnode"],
                        count_bx : current_state.values["count"],
                        revision_bx : current_state.values["revision_number"],
                        nnode_bx : current_state.next,
                        threadid_bx : self.thread_id,
                        thread_pd : gr.Dropdown(label="选择线程", choices=self.threads, value=self.thread_id,interactive=True),
                        step_pd : gr.Dropdown(label="回退到: thread:count:last_node:next_node:rev:thread_ts", 
                               choices=hist, value=hist[0],interactive=True),
                    }
            def get_snapshots():
                new_label = f"thread_id: {self.thread_id}, Summary of snapshots"
                sstate = ""
                for state in self.graph.get_state_history(self.thread):
                    for key in ['plan', 'draft', 'critique']:
                        if key in state.values:
                            state.values[key] = state.values[key][:80] + "..."
                    if 'content' in state.values:
                        for i in range(len(state.values['content'])):
                            state.values['content'][i] = state.values['content'][i][:20] + '...'
                    if 'writes' in state.metadata:
                        state.metadata['writes'] = "not shown"
                    sstate += str(state) + "\n\n"
                return gr.update(label=new_label, value=sstate)

            def vary_btn(stat):
                #print(f"vary_btn{stat}")
                return(gr.update(variant=stat))
            
            with gr.Tab("写作助手"):
                with gr.Row():
                    topic_bx = gr.Textbox(label="作文主题", value="白鹭")
                    gen_btn = gr.Button("开始", scale=0,min_width=80, variant='primary')
                    cont_btn = gr.Button("继续", scale=0,min_width=80)
                with gr.Row():
                    lnode_bx = gr.Textbox(label="上一步骤", min_width=100)
                    nnode_bx = gr.Textbox(label="下一步骤", min_width=100)
                    threadid_bx = gr.Textbox(label="线程", scale=0, min_width=80)
                    revision_bx = gr.Textbox(label="版本号", scale=0, min_width=80)
                    count_bx = gr.Textbox(label="执行步数", scale=0, min_width=80)
                with gr.Accordion("管理", open=False):
                    checks = list(self.graph.nodes.keys())
                    checks.remove('__start__')
                    stop_after = gr.CheckboxGroup(checks,label="执行完毕后中断", value=checks, scale=0, min_width=400)
                    with gr.Row():
                        thread_pd = gr.Dropdown(choices=self.threads,interactive=True, label="选择线程", min_width=120, scale=0)
                        step_pd = gr.Dropdown(choices=['N/A'],interactive=True, label="选择步骤", min_width=160, scale=1)
                live = gr.Textbox(label="实时输出", lines=5, max_lines=5)
        
                # actions
                sdisps =[topic_bx,lnode_bx,nnode_bx,threadid_bx,revision_bx,count_bx,step_pd,thread_pd]
                thread_pd.input(self.switch_thread, [thread_pd], None).then(
                                fn=updt_disp, inputs=None, outputs=sdisps)
                step_pd.input(self.copy_state,[step_pd],None).then(
                              fn=updt_disp, inputs=None, outputs=sdisps)
                gen_btn.click(vary_btn,gr.Number("secondary", visible=False), gen_btn).then(
                              fn=self.run_agent, inputs=[gr.Number(True, visible=False),topic_bx,stop_after], outputs=[live],show_progress=True).then(
                              fn=updt_disp, inputs=None, outputs=sdisps).then( 
                              vary_btn,gr.Number("primary", visible=False), gen_btn).then(
                              vary_btn,gr.Number("primary", visible=False), cont_btn)
                cont_btn.click(vary_btn,gr.Number("secondary", visible=False), cont_btn).then(
                               fn=self.run_agent, inputs=[gr.Number(False, visible=False),topic_bx,stop_after], 
                               outputs=[live]).then(
                               fn=updt_disp, inputs=None, outputs=sdisps).then(
                               vary_btn,gr.Number("primary", visible=False), cont_btn)
        
            with gr.Tab("大纲"):
                with gr.Row():
                    refresh_btn = gr.Button("刷新")
                    modify_btn = gr.Button("修改")
                plan = gr.Textbox(label="大纲", lines=10, interactive=True)
                refresh_btn.click(fn=self.get_state, inputs=gr.Number("plan", visible=False), outputs=plan)
                modify_btn.click(fn=self.modify_state, inputs=[gr.Number("plan", visible=False),
                                                          gr.Number("planner", visible=False), plan],outputs=None).then(
                                 fn=updt_disp, inputs=None, outputs=sdisps)
            with gr.Tab("调研结果"):
                refresh_btn = gr.Button("刷新")
                content_bx = gr.Textbox(label="内容", lines=10)
                refresh_btn.click(fn=self.get_content, inputs=None, outputs=content_bx)
            with gr.Tab("草稿"):
                with gr.Row():
                    refresh_btn = gr.Button("刷新")
                    modify_btn = gr.Button("修改")
                draft_bx = gr.Textbox(label="草稿", lines=10, interactive=True)
                refresh_btn.click(fn=self.get_state, inputs=gr.Number("draft", visible=False), outputs=draft_bx)
                modify_btn.click(fn=self.modify_state, inputs=[gr.Number("draft", visible=False),
                                                          gr.Number("generate", visible=False), draft_bx], outputs=None).then(
                                fn=updt_disp, inputs=None, outputs=sdisps)
            with gr.Tab("改进意见"):
                with gr.Row():
                    refresh_btn = gr.Button("刷新")
                    modify_btn = gr.Button("修改")
                critique_bx = gr.Textbox(label="改进意见", lines=10, interactive=True)
                refresh_btn.click(fn=self.get_state, inputs=gr.Number("critique", visible=False), outputs=critique_bx)
                modify_btn.click(fn=self.modify_state, inputs=[gr.Number("critique", visible=False),
                                                          gr.Number("reflect", visible=False), 
                                                          critique_bx], outputs=None).then(
                                fn=updt_disp, inputs=None, outputs=sdisps)
            with gr.Tab("状态快照"):
                with gr.Row():
                    refresh_btn = gr.Button("刷新")
                snapshots = gr.Textbox(label="状态快照")
                refresh_btn.click(fn=get_snapshots, inputs=None, outputs=snapshots)
        return demo

    def launch(self, share=None):
        if port := os.getenv("PORT1"):
            self.demo.launch(share=True, server_port=int(port), server_name="0.0.0.0")
        else:
            self.demo.launch(share=self.share)

if __name__ == '__main__':
    MultiAgent = writer()
    app = writer_gui(MultiAgent.graph)
    app.launch()