//! Golden and differential tests for the Jinja subset engine.
//!
//! Three independent sources of truth:
//!
//! 1. `BONSAI2_CHAT_TEMPLATE` is the verbatim `tokenizer.chat_template` of
//!    PrismML Bonsai 2 27B, and `GOLDEN_CASES` are the five prompts the
//!    PrismML llama.cpp fork produced from it through `POST /apply-template`.
//!    Rendering must be **byte-identical**.
//! 2. `EXTRA_TEMPLATE_CASES`, `RAISE_CASES` and `MICRO_CASES` were generated
//!    by running the same template (and 100+ focused micro-templates) through
//!    Python `jinja2` 3.1 configured exactly like HuggingFace `transformers`
//!    (`trim_blocks`, `lstrip_blocks`, sandboxed), with `tojson` replaced by
//!    plain `json.dumps` defaults — which is what the fork's renderer emits.
//!    They cover the template branches the five goldens never reach: tool
//!    calls, tool responses, vision parts, `preserve_thinking`, and every
//!    `raise_exception` path.
//! 3. The negative tests below assert that anything outside the supported
//!    subset is a hard error with a source position — the actual finding
//!    behind this engine (TOK-07: the old engine rendered garbage instead).

use oxibonsai_tokenizer::jinja::{JinjaError, JinjaOptions, JinjaTemplate, Value, ValueMap};
use proptest::prelude::*;

/// The verbatim `tokenizer.chat_template` of PrismML Bonsai 2 27B.
const BONSAI2_CHAT_TEMPLATE: &str = r##"{%- set image_count = namespace(value=0) %}
{%- set video_count = namespace(value=0) %}
{%- macro render_content(content, do_vision_count, is_system_content=false) %}
    {%- if content is string %}
        {{- content }}
    {%- elif content is iterable and content is not mapping %}
        {%- for item in content %}
            {%- if 'image' in item or 'image_url' in item or item.type == 'image' %}
                {%- if is_system_content %}
                    {{- raise_exception('System message cannot contain images.') }}
                {%- endif %}
                {%- if do_vision_count %}
                    {%- set image_count.value = image_count.value + 1 %}
                {%- endif %}
                {%- if add_vision_id %}
                    {{- 'Picture ' ~ image_count.value ~ ': ' }}
                {%- endif %}
                {{- '<|vision_start|><|image_pad|><|vision_end|>' }}
            {%- elif 'video' in item or item.type == 'video' %}
                {%- if is_system_content %}
                    {{- raise_exception('System message cannot contain videos.') }}
                {%- endif %}
                {%- if do_vision_count %}
                    {%- set video_count.value = video_count.value + 1 %}
                {%- endif %}
                {%- if add_vision_id %}
                    {{- 'Video ' ~ video_count.value ~ ': ' }}
                {%- endif %}
                {{- '<|vision_start|><|video_pad|><|vision_end|>' }}
            {%- elif 'text' in item %}
                {{- item.text }}
            {%- else %}
                {{- raise_exception('Unexpected item type in content.') }}
            {%- endif %}
        {%- endfor %}
    {%- elif content is none or content is undefined %}
        {{- '' }}
    {%- else %}
        {{- raise_exception('Unexpected content type.') }}
    {%- endif %}
{%- endmacro %}
{%- if not messages %}
    {{- raise_exception('No messages provided.') }}
{%- endif %}
{%- set reasoning_instructions = '' %}
{%- if enable_thinking is undefined or enable_thinking is true %}
    {%- set resolved_reasoning_effort = reasoning_effort|default('xhigh') %}
    {%- if resolved_reasoning_effort not in ('xhigh', 'medium', 'low') %}
        {{- raise_exception('Unexpected reasoning effort ' ~ reasoning_effort ~ '. Supported types are xhigh (default), medium, and low.') }}
    {%- endif %}
    {%- if resolved_reasoning_effort == 'xhigh' %}
        {%- set reasoning_instructions = 'Reasoning effort is set to xhigh. Please think carefully through the task, validate key assumptions, consider plausible alternatives, and prioritize correctness, consistency, and clarity in the final answer.' %}
    {%- elif resolved_reasoning_effort == 'low' %}
        {%- set reasoning_instructions = 'Reasoning effort is set to low. Keep your thinking brief and focused, moving directly to the conclusion without unnecessary elaboration.' %}
    {%- endif %}
{%- endif %}
{%- if tools and tools is iterable and tools is not mapping %}
    {{- '<|im_start|>system\n' }}
    {%- if reasoning_instructions %}
        {{- reasoning_instructions + '\n\n' }}
    {%- endif %}
    {{- "# Tools\n\nYou have access to the following functions:\n\n<tools>" }}
    {%- for tool in tools %}
        {{- "\n" }}
        {{- tool | tojson }}
    {%- endfor %}
    {{- "\n</tools>" }}
    {{- '\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n</parameter>\n<parameter=example_parameter_2>\nThis is the value for the second parameter\nthat can span\nmultiple lines\n</parameter>\n</function>\n</tool_call>\n\n<IMPORTANT>\nReminder:\n- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags\n- Required parameters MUST be specified\n- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after\n- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls\n</IMPORTANT>' }}
    {%- if messages[0].role == 'system' %}
        {%- set content = render_content(messages[0].content, false, true)|trim %}
        {%- if content %}
            {{- '\n\n' + content }}
        {%- endif %}
    {%- endif %}
    {{- '<|im_end|>\n' }}
{%- else %}
    {%- if messages[0].role == 'system' %}
        {%- set content = render_content(messages[0].content, false, true)|trim %}
        {%- if content %}
            {{- '<|im_start|>system\n' + (reasoning_instructions + '\n\n' if reasoning_instructions else '')  + content + '<|im_end|>\n' }}
        {%- elif reasoning_instructions %}
            {{- '<|im_start|>system\n' + reasoning_instructions + '<|im_end|>\n' }}
        {%- endif %}
    {%- elif reasoning_instructions %}
        {{- '<|im_start|>system\n' + reasoning_instructions + '<|im_end|>\n' }}
    {%- endif %}
{%- endif %}
{%- set ns = namespace(multi_step_tool=true, last_query_index=messages|length - 1) %}
{%- for message in messages[::-1] %}
    {%- set index = (messages|length - 1) - loop.index0 %}
    {%- if ns.multi_step_tool and message.role == "user" %}
        {%- set content = render_content(message.content, false)|trim %}
        {%- if not(content.startswith('<tool_response>') and content.endswith('</tool_response>')) %}
            {%- set ns.multi_step_tool = false %}
            {%- set ns.last_query_index = index %}
        {%- endif %}
    {%- endif %}
{%- endfor %}
{%- if ns.multi_step_tool %}
    {{- raise_exception('No user query found in messages.') }}
{%- endif %}
{%- for message in messages %}
    {%- set content = render_content(message.content, true)|trim %}
    {%- if message.role == "system" %}
        {%- if not loop.first %}
            {{- raise_exception('System message must be at the beginning.') }}
        {%- endif %}
    {%- elif message.role == "user" %}
        {{- '<|im_start|>' + message.role + '\n' + content + '<|im_end|>' + '\n' }}
    {%- elif message.role == "assistant" %}
        {%- set reasoning_content = '' %}
        {%- if message.reasoning_content is string %}
            {%- set reasoning_content = message.reasoning_content %}
        {%- endif %}
        {%- set reasoning_content = reasoning_content|trim %}
        {%- if preserve_thinking is undefined or preserve_thinking is true or loop.index0 > ns.last_query_index %}
            {{- '<|im_start|>' + message.role + '\n<think>\n' + reasoning_content + '\n</think>\n\n' + content }}
        {%- else %}
            {{- '<|im_start|>' + message.role + '\n' + content }}
        {%- endif %}
        {%- if message.tool_calls and message.tool_calls is iterable and message.tool_calls is not mapping %}
            {%- for tool_call in message.tool_calls %}
                {%- if tool_call.function is defined %}
                    {%- set tool_call = tool_call.function %}
                {%- endif %}
                {%- if loop.first %}
                    {%- if content|trim %}
                        {{- '\n\n<tool_call>\n<function=' + tool_call.name + '>\n' }}
                    {%- else %}
                        {{- '<tool_call>\n<function=' + tool_call.name + '>\n' }}
                    {%- endif %}
                {%- else %}
                    {{- '\n<tool_call>\n<function=' + tool_call.name + '>\n' }}
                {%- endif %}
                {%- if tool_call.arguments is defined and tool_call.arguments != '' %}
                    {%- for args_name, args_value in tool_call.arguments|items %}
                        {{- '<parameter=' + args_name + '>\n' }}
                        {%- set args_value = args_value | string if args_value is string else args_value | tojson | safe %}
                        {{- args_value }}
                        {{- '\n</parameter>\n' }}
                    {%- endfor %}
                {%- endif %}
                {{- '</function>\n</tool_call>' }}
            {%- endfor %}
        {%- endif %}
        {{- '<|im_end|>\n' }}
    {%- elif message.role == "tool" %}
        {%- if loop.previtem and loop.previtem.role != "tool" %}
            {{- '<|im_start|>user' }}
        {%- endif %}
        {{- '\n<tool_response>\n' }}
        {{- content }}
        {{- '\n</tool_response>' }}
        {%- if not loop.last and loop.nextitem.role != "tool" %}
            {{- '<|im_end|>\n' }}
        {%- elif loop.last %}
            {{- '<|im_end|>\n' }}
        {%- endif %}
    {%- else %}
        {{- raise_exception('Unexpected message role.') }}
    {%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}
    {{- '<|im_start|>assistant\n' }}
    {%- if enable_thinking is defined and enable_thinking is false %}
        {{- '<think>\n\n</think>\n\n' }}
    {%- else %}
        {{- '<think>\n' }}
    {%- endif %}
{%- endif %}"##;

// Generated tables: the fork's golden set plus a Python jinja2 oracle run
// (see the module docs above). Regenerate rather than hand-editing.
pub const GOLDEN_CASES: &[(&str, &str, &str)] = &[
    ("case0", "{\"messages\": [{\"role\": \"user\", \"content\": \"What is 2+2? Answer briefly.\"}], \"add_generation_prompt\": true}", "<|im_start|>system\nReasoning effort is set to xhigh. Please think carefully through the task, validate key assumptions, consider plausible alternatives, and prioritize correctness, consistency, and clarity in the final answer.<|im_end|>\n<|im_start|>user\nWhat is 2+2? Answer briefly.<|im_end|>\n<|im_start|>assistant\n<think>\n"),
    ("case1", "{\"messages\": [{\"role\": \"user\", \"content\": \"What is 2+2?\"}], \"enable_thinking\": false, \"add_generation_prompt\": true}", "<|im_start|>user\nWhat is 2+2?<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"),
    ("case2", "{\"messages\": [{\"role\": \"user\", \"content\": \"What is 2+2?\"}], \"reasoning_effort\": \"low\", \"add_generation_prompt\": true}", "<|im_start|>system\nReasoning effort is set to low. Keep your thinking brief and focused, moving directly to the conclusion without unnecessary elaboration.<|im_end|>\n<|im_start|>user\nWhat is 2+2?<|im_end|>\n<|im_start|>assistant\n<think>\n"),
    ("case3", "{\"messages\": [{\"role\": \"system\", \"content\": \"You are a helpful assistant\"}, {\"role\": \"user\", \"content\": \"Hi\"}, {\"role\": \"assistant\", \"content\": \"Hello!\", \"reasoning_content\": \"user greets\"}, {\"role\": \"user\", \"content\": \"Bye\"}], \"add_generation_prompt\": true}", "<|im_start|>system\nReasoning effort is set to xhigh. Please think carefully through the task, validate key assumptions, consider plausible alternatives, and prioritize correctness, consistency, and clarity in the final answer.\n\nYou are a helpful assistant<|im_end|>\n<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n<think>\nuser greets\n</think>\n\nHello!<|im_end|>\n<|im_start|>user\nBye<|im_end|>\n<|im_start|>assistant\n<think>\n"),
    ("case4", "{\"messages\": [{\"role\": \"user\", \"content\": \"weather?\"}], \"tools\": [{\"type\": \"function\", \"function\": {\"name\": \"get_weather\", \"description\": \"Get weather\", \"parameters\": {\"type\": \"object\", \"properties\": {\"city\": {\"type\": \"string\"}}, \"required\": [\"city\"]}}}], \"add_generation_prompt\": true}", "<|im_start|>system\nReasoning effort is set to xhigh. Please think carefully through the task, validate key assumptions, consider plausible alternatives, and prioritize correctness, consistency, and clarity in the final answer.\n\n# Tools\n\nYou have access to the following functions:\n\n<tools>\n{\"type\": \"function\", \"function\": {\"name\": \"get_weather\", \"description\": \"Get weather\", \"parameters\": {\"type\": \"object\", \"properties\": {\"city\": {\"type\": \"string\"}}, \"required\": [\"city\"]}}}\n</tools>\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n</parameter>\n<parameter=example_parameter_2>\nThis is the value for the second parameter\nthat can span\nmultiple lines\n</parameter>\n</function>\n</tool_call>\n\n<IMPORTANT>\nReminder:\n- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags\n- Required parameters MUST be specified\n- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after\n- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls\n</IMPORTANT><|im_end|>\n<|im_start|>user\nweather?<|im_end|>\n<|im_start|>assistant\n<think>\n"),
];

pub const MICRO_CASES: &[(&str, &str, &str, &str)] = &[
    ("filter_binds_tighter_than_ternary_str", "{{ v | string if v is string else v | tojson | safe }}", "{\"v\": \"raw text\"}", "raw text"),
    ("filter_binds_tighter_than_ternary_obj", "{{ v | string if v is string else v | tojson | safe }}", "{\"v\": {\"b\": 1, \"a\": [1, 2]}}", "{\"b\": 1, \"a\": [1, 2]}"),
    ("concat_binds_tighter_than_compare", "{{ 'a' ~ 'b' == 'ab' }}", "{}", "True"),
    ("concat_binds_tighter_than_plus", "{{ 'a' ~ 'b' + 'c' ~ 'd' }}", "{}", "abcd"),
    ("concat_stringifies", "{{ 'n=' ~ 1 ~ true ~ none }}", "{}", "n=1TrueNone"),
    ("mul_binds_tighter_than_plus", "{{ 1 + 2 * 3 - 4 // 3 }}", "{}", "6"),
    ("pow_and_mod", "{{ 2 ** 10 }}|{{ 17 % 5 }}|{{ 7 / 2 }}", "{}", "1024|2|3.5"),
    ("unary_minus", "{{ -3 + 1 }}|{{ not false }}|{{ not 0 }}", "{}", "-2|True|True"),
    ("chained_compare", "{{ 1 < 2 }}|{{ 3 <= 3 }}|{{ 'a' < 'b' }}|{{ 2 != 3 }}", "{}", "True|True|True|True"),
    ("and_or_not_precedence", "{{ true or false and false }}|{{ not true or true }}", "{}", "True|True"),
    ("inline_if_without_else", "[{{ 'y' if flag }}][{{ 'y' if not flag }}]", "{\"flag\": true}", "[y][]"),
    ("paren_grouping", "{{ (1 + 2) * 3 }}", "{}", "9"),
    ("not_of_parenthesised_and", "{{ not(s.startswith('<a>') and s.endswith('</a>')) }}", "{\"s\": \"<a>x</a>\"}", "False"),
    ("slice_reverse", "{% for m in xs[::-1] %}{{ m }}{% endfor %}", "{\"xs\": [\"a\", \"b\", \"c\", \"d\"]}", "dcba"),
    ("slice_from", "{{ xs[1:] | join(',') }}", "{\"xs\": [\"a\", \"b\", \"c\", \"d\"]}", "b,c,d"),
    ("slice_to", "{{ xs[:2] | join(',') }}", "{\"xs\": [\"a\", \"b\", \"c\", \"d\"]}", "a,b"),
    ("slice_step", "{{ xs[::2] | join(',') }}", "{\"xs\": [\"a\", \"b\", \"c\", \"d\", \"e\"]}", "a,c,e"),
    ("slice_negative_bounds", "{{ xs[-3:-1] | join(',') }}", "{\"xs\": [\"a\", \"b\", \"c\", \"d\", \"e\"]}", "c,d"),
    ("slice_string", "{{ s[1:4] }}|{{ s[::-1] }}", "{\"s\": \"abcdef\"}", "bcd|fedcba"),
    ("index_negative", "{{ xs[-1] }}|{{ xs[0] }}", "{\"xs\": [\"a\", \"b\", \"c\"]}", "c|a"),
    ("attr_and_item_equivalence", "{{ d.a }}|{{ d['a'] }}|{{ d.missing is undefined }}", "{\"d\": {\"a\": 1}}", "1|1|True"),
    ("loop_fields", "{% for x in xs %}{{ loop.index0 }}/{{ loop.index }}/{{ loop.revindex }}/{{ loop.revindex0 }}/{{ loop.first }}/{{ loop.last }}/{{ loop.length }};{% endfor %}", "{\"xs\": [\"a\", \"b\", \"c\"]}", "0/1/3/2/True/False/3;1/2/2/1/False/False/3;2/3/1/0/False/True/3;"),
    ("loop_previtem_nextitem", "{% for x in xs %}[{{ loop.previtem }}|{{ x }}|{{ loop.nextitem }}]{% endfor %}", "{\"xs\": [\"a\", \"b\", \"c\"]}", "[|a|b][a|b|c][b|c|]"),
    ("loop_nested", "{% for a in xs %}{% for b in ys %}{{ a }}{{ b }}{{ loop.index }};{% endfor %}{{ loop.index }}|{% endfor %}", "{\"xs\": [\"a\", \"b\"], \"ys\": [1, 2]}", "a11;a22;1|b11;b22;2|"),
    ("loop_else_taken", "{% for x in xs %}{{ x }}{% else %}EMPTY{% endfor %}", "{\"xs\": []}", "EMPTY"),
    ("loop_else_skipped", "{% for x in xs %}{{ x }}{% else %}EMPTY{% endfor %}", "{\"xs\": [\"z\"]}", "z"),
    ("loop_filtered", "{% for x in xs if x != 'b' %}{{ x }}{{ loop.index }}{% endfor %}", "{\"xs\": [\"a\", \"b\", \"c\"]}", "a1c2"),
    ("loop_unpack_pairs", "{% for k, v in d | items %}{{ k }}={{ v }};{% endfor %}", "{\"d\": {\"z\": 1, \"a\": \"two\"}}", "z=1;a=two;"),
    ("loop_over_string", "{% for c in s %}{{ c }}.{% endfor %}", "{\"s\": \"abc\"}", "a.b.c."),
    ("loop_over_mapping_keys", "{% for k in d %}{{ k }};{% endfor %}", "{\"d\": {\"b\": 1, \"a\": 2}}", "b;a;"),
    ("loop_break_continue", "{% for x in xs %}{% if x == 'b' %}{% continue %}{% endif %}{% if x == 'd' %}{% break %}{% endif %}{{ x }}{% endfor %}", "{\"xs\": [\"a\", \"b\", \"c\", \"d\", \"e\"]}", "ac"),
    ("set_inside_if_escapes", "{% set x = 'a' %}{% if true %}{% set x = 'b' %}{% endif %}{{ x }}", "{}", "b"),
    ("set_inside_for_is_local", "{% set x = 'a' %}{% for i in xs %}{% set x = 'b' %}{% endfor %}{{ x }}", "{\"xs\": [1]}", "a"),
    ("namespace_mutation_in_loop", "{% set ns = namespace(n=0, seen='') %}{% for i in xs %}{% set ns.n = ns.n + i %}{% set ns.seen = ns.seen ~ i %}{% endfor %}{{ ns.n }}|{{ ns.seen }}", "{\"xs\": [1, 2, 3]}", "6|123"),
    ("set_tuple_targets", "{% set a, b = 1, 'two' %}{{ a }}|{{ b }}", "{}", "1|two"),
    ("set_block", "{% set body %}hello {{ who }}{% endset %}[{{ body }}]", "{\"who\": \"world\"}", "[hello world]"),
    ("set_attr_on_namespace_nested", "{% set ns = namespace(v=namespace(w=1)) %}{{ ns.v.w }}", "{}", "1"),
    ("tests_core", "{{ s is string }}|{{ s is iterable }}|{{ s is mapping }}|{{ d is mapping }}|{{ l is iterable }}|{{ n is none }}|{{ u is undefined }}|{{ s is defined }}|{{ t is true }}|{{ f is false }}|{{ i is number }}|{{ i is integer }}|{{ t is boolean }}|{{ l is sequence }}", "{\"s\": \"x\", \"d\": {\"a\": 1}, \"l\": [1], \"n\": null, \"t\": true, \"f\": false, \"i\": 3}", "True|True|False|True|True|True|True|True|True|True|True|True|True|True"),
    ("tests_negated", "{{ d is not mapping }}|{{ s is not string }}|{{ n is not none }}", "{\"s\": \"x\", \"d\": {\"a\": 1}, \"n\": null}", "False|False|False"),
    ("undefined_is_falsy_and_renders_empty", "[{{ u }}]{% if u %}T{% else %}F{% endif %}{{ u is defined }}", "{}", "[]FFalse"),
    ("membership", "{{ 'a' in l }}|{{ 'z' in l }}|{{ 'k' in d }}|{{ 'v' in d }}|{{ 'bc' in s }}|{{ 'x' not in l }}|{{ e not in ('xhigh', 'medium', 'low') }}", "{\"l\": [\"a\", \"b\"], \"d\": {\"k\": \"v\"}, \"s\": \"abcd\", \"e\": \"nope\"}", "True|False|True|False|True|True|True"),
    ("literals", "{{ [1, 'a', true, none] }}|{{ {'b': 1, 'a': 2} }}|{{ (1, 2) }}|{{ 1.5 }}|{{ 10 }}", "{}", "[1, 'a', True, None]|{'b': 1, 'a': 2}|(1, 2)|1.5|10"),
    ("string_escapes", "{{ 'a\\nb' }}|{{ \"q\\\"q\" }}|{{ 'ta\\tb' }}", "{}", "a\nb|q\"q|ta\tb"),
    ("filter_trim", "[{{ s | trim }}]", "{\"s\": \"  padded \\n\"}", "[padded]"),
    ("filter_length", "{{ s | length }}|{{ l | length }}|{{ d | length }}", "{\"s\": \"abcd\", \"l\": [1, 2], \"d\": {\"a\": 1}}", "4|2|1"),
    ("filter_replace", "{{ s | replace('a', 'X') }}|{{ s | replace('a', 'X', 1) }}", "{\"s\": \"banana\"}", "bXnXnX|bXnana"),
    ("filter_join", "{{ l | join('-') }}|{{ l | join }}", "{\"l\": [\"a\", \"b\", \"c\"]}", "a-b-c|abc"),
    ("filter_join_ints", "{{ l | join(', ') }}", "{\"l\": [1, 2, 3]}", "1, 2, 3"),
    ("filter_first_last", "{{ l | first }}|{{ l | last }}|[{{ [] | first }}]", "{\"l\": [\"a\", \"b\"]}", "a|b|[]"),
    ("filter_list", "{{ s | list | join('-') }}|{{ d | list | join('-') }}", "{\"s\": \"abc\", \"d\": {\"b\": 1, \"a\": 2}}", "a-b-c|b-a"),
    ("filter_default", "{{ u | default('D') }}|{{ v | default('D') }}|{{ '' | default('D', true) }}", "{\"v\": \"set\"}", "D|set|D"),
    ("filter_string_int", "{{ 12 | string }}|{{ '13' | int }}|{{ 'nope' | int }}|{{ '2.7' | int }}|{{ 2.7 | int }}", "{}", "12|13|0|2|2"),
    ("filter_lower_upper", "{{ s | upper }}|{{ s | lower }}", "{\"s\": \"MiXeD\"}", "MIXED|mixed"),
    ("filter_items", "{% for k, v in d | items %}{{ k }}:{{ v }};{% endfor %}", "{\"d\": {\"z\": 1, \"a\": 2}}", "z:1;a:2;"),
    ("filter_selectattr_truthy", "{% for x in l | selectattr('on') %}{{ x.n }};{% endfor %}", "{\"l\": [{\"n\": \"a\", \"on\": true}, {\"n\": \"b\", \"on\": false}, {\"n\": \"c\", \"on\": true}]}", "a;c;"),
    ("filter_selectattr_equalto", "{% for x in l | selectattr('role', 'equalto', 'user') %}{{ x.n }};{% endfor %}", "{\"l\": [{\"n\": \"a\", \"role\": \"user\"}, {\"n\": \"b\", \"role\": \"tool\"}]}", "a;"),
    ("filter_selectattr_defined", "{% for x in l | selectattr('role', 'defined') %}{{ x.n }};{% endfor %}", "{\"l\": [{\"n\": \"a\", \"role\": \"user\"}, {\"n\": \"b\"}]}", "a;"),
    ("filter_map_attribute", "{{ l | map(attribute='n') | join(',') }}", "{\"l\": [{\"n\": \"a\"}, {\"n\": \"b\"}]}", "a,b"),
    ("filter_map_filter", "{{ l | map('upper') | join(',') }}", "{\"l\": [\"a\", \"b\"]}", "A,B"),
    ("filter_safe_is_identity", "{{ s | safe }}", "{\"s\": \"<b>&amp;</b>\"}", "<b>&amp;</b>"),
    ("filter_chain", "{{ s | trim | upper | replace(' ', '_') }}", "{\"s\": \"  a b  \"}", "A_B"),
    ("tojson_nested_tool", "{{ t | tojson }}", "{\"t\": {\"type\": \"function\", \"function\": {\"name\": \"get_weather\", \"description\": \"Get weather\", \"parameters\": {\"type\": \"object\", \"properties\": {\"city\": {\"type\": \"string\"}}, \"required\": [\"city\"]}}}}", "{\"type\": \"function\", \"function\": {\"name\": \"get_weather\", \"description\": \"Get weather\", \"parameters\": {\"type\": \"object\", \"properties\": {\"city\": {\"type\": \"string\"}}, \"required\": [\"city\"]}}}"),
    ("tojson_scalars", "{{ a | tojson }}|{{ b | tojson }}|{{ c | tojson }}|{{ d | tojson }}", "{\"a\": null, \"b\": true, \"c\": 17, \"d\": -2}", "null|true|17|-2"),
    ("tojson_empty", "{{ a | tojson }}|{{ b | tojson }}|{{ c | tojson }}", "{\"a\": [], \"b\": {}, \"c\": \"\"}", "[]|{}|\"\""),
    ("tojson_insertion_order", "{{ d | tojson }}", "{\"d\": {\"z\": 1, \"m\": 2, \"a\": 3}}", "{\"z\": 1, \"m\": 2, \"a\": 3}"),
    ("macro_with_defaults", "{% macro m(a, b, c=false) %}{{ a }}/{{ b }}/{{ c }}{% endmacro %}{{ m(1, 2) }}|{{ m(1, 2, true) }}|{{ m(b=4, a=3) }}", "{}", "1/2/False|1/2/True|3/4/False"),
    ("macro_result_filtered", "{% macro m(x) %}  {{ x }}  {% endmacro %}[{{ m('v') | trim }}]", "{}", "[v]"),
    ("macro_sees_globals", "{% macro m() %}{{ g }}{% endmacro %}{% set g = 'G' %}{{ m() }}", "{}", "G"),
    ("macro_recursive_render_content", "{% macro rc(content) %}{%- if content is string %}{{- content }}{%- elif content is iterable and content is not mapping %}{%- for item in content %}{%- if 'text' in item %}{{- item.text }}{%- endif %}{%- endfor %}{%- elif content is none or content is undefined %}{{- '' }}{%- endif %}{% endmacro %}[{{ rc('plain') | trim }}][{{ rc([{'text': 'a'}, {'text': 'b'}]) | trim }}][{{ rc(none) | trim }}]", "{}", "[plain][ab][]"),
    ("call_block", "{% macro wrap() %}<{{ caller() }}>{% endmacro %}{% call wrap() %}inner{% endcall %}", "{}", "<inner>"),
    ("call_block_with_args", "{% macro each(xs) %}{% for x in xs %}{{ caller(x) }}{% endfor %}{% endmacro %}{% call(item) each(['a','b']) %}[{{ item }}]{% endcall %}", "{}", "[a][b]"),
    ("string_methods", "{{ s.startswith('ab') }}|{{ s.endswith('ef') }}|{{ s.upper() }}|{{ s.lower() }}|{{ p.strip() }}|{{ s.replace('b', 'B') }}|{{ 'a,b' .split(',') | join('-') }}", "{\"s\": \"abcdef\", \"p\": \"  x  \"}", "True|True|ABCDEF|abcdef|x|aBcdef|a-b"),
    ("mapping_methods", "{{ d.get('a') }}|{{ d.get('zz') }}|{{ d.keys() | list | join(',') }}", "{\"d\": {\"a\": 1, \"b\": 2}}", "1|None|a,b"),
    ("comment_is_dropped", "a{# this is\na comment #}b", "{}", "ab"),
    ("whitespace_trim_and_lstrip_blocks", "start\n    {% if flag %}\n  kept\n    {% endif %}\nend", "{\"flag\": true}", "start\n  kept\nend"),
    ("whitespace_minus_markers", "start\n    {%- if flag %}\n  kept\n    {%- endif %}\nend", "{\"flag\": true}", "start  keptend"),
    ("whitespace_output_markers", "a    {{- v -}}    b", "{\"v\": \"V\"}", "aVb"),
    ("trailing_newline_dropped", "line\n", "{}", "line"),
    ("trailing_newlines_only_one_dropped", "line\n\n", "{}", "line\n"),
    ("if_elif_else", "{% if v == 1 %}one{% elif v == 2 %}two{% elif v == 3 %}three{% else %}other{% endif %}", "{\"v\": 3}", "three"),
    ("if_else_false_branch", "{% if v %}T{% else %}F{% endif %}", "{\"v\": 0}", "F"),
    ("output_scalar_repr", "{{ t }}|{{ f }}|{{ n }}|{{ i }}|{{ fl }}|{{ s }}", "{\"t\": true, \"f\": false, \"n\": null, \"i\": -5, \"fl\": 1.5, \"s\": \"s\"}", "True|False|None|-5|1.5|s"),
    ("chained_compare_python", "{{ 1 < 2 < 3 }}|{{ 3 < 2 < 1 }}", "{}", "True|False"),
    ("float_repr", "{{ a }}|{{ b }}|{{ c }}|{{ a | tojson }}|{{ c | tojson }}", "{\"a\": 1.0, \"b\": 2.5, \"c\": -0.0}", "1.0|2.5|-0.0|1.0|-0.0"),
    ("tojson_floats", "{{ d | tojson }}", "{\"d\": {\"a\": 1.0, \"b\": 2.5, \"c\": 0.1}}", "{\"a\": 1.0, \"b\": 2.5, \"c\": 0.1}"),
    ("tojson_specials", "{{ s | tojson }}", "{\"s\": \"< > & ' \\\" \\\\ \\n \\t \\u007f \\u00e9 \\ud83d\\ude00\"}", "\"< > & ' \\\" \\\\ \\n \\t \\u007f \\u00e9 \\ud83d\\ude00\""),
    ("str_repetition", "{{ 'ab' * 3 }}|{{ 3 * 'x' }}", "{}", "ababab|xxx"),
    ("list_concat", "{{ [1] + [2] }}", "{}", "[1, 2]"),
    ("missing_key_and_index_are_undefined", "[{{ d['missing'] }}][{{ xs[10] }}][{{ n.attr }}]", "{\"d\": {}, \"xs\": [1], \"n\": null}", "[][][]"),
    ("undefined_coercions", "[{{ u | trim }}][{{ u | length }}][{{ 'a' ~ u }}][{{ u == 'x' }}][{{ u != 'x' }}]", "{}", "[][0][a][False][True]"),
    ("iterate_undefined_is_empty", "[{% for x in u %}{{ x }}{% endfor %}]", "{}", "[]"),
    ("macro_missing_arg_is_undefined", "{% macro m(a, b) %}[{{ a }}{{ b }}]{% endmacro %}{{ m(1) }}", "{}", "[1]"),
    ("namespace_missing_field_is_undefined", "{% set n = namespace(a=1) %}{{ n.a }}|{{ n.b is undefined }}", "{}", "1|True"),
    ("dict_literal_duplicate_key", "{{ {'a': 1, 'a': 2} | tojson }}", "{}", "{\"a\": 2}"),
    ("bool_int_equality", "{{ true == 1 }}|{{ false == 0 }}", "{}", "True|True"),
    ("join_stringifies", "{{ [1, none, true] | join('-') }}", "{}", "1-None-True"),
    ("map_missing_attribute", "[{{ l | map(attribute='zz') | list | join(',') }}]", "{\"l\": [{\"a\": 1}]}", "[]"),
    ("selectattr_missing_attribute", "{{ l | selectattr('zz') | list | length }}", "{\"l\": [{\"a\": 1}]}", "0"),
    ("default_does_not_replace_none", "{{ n | default('D') }}", "{\"n\": null}", "None"),
    ("trim_strips_unicode_whitespace", "[{{ s | trim }}]", "{\"s\": \"\\u00a0 x \\u00a0\"}", "[x]"),
    ("replace_empty_pattern", "{{ 'abc' | replace('', '-') }}", "{}", "-a-b-c-"),
    ("string_of_mapping", "{{ d | string }}", "{\"d\": {\"a\": 1}}", "{'a': 1}"),
    ("upper_coerces_non_strings", "{{ 3 | upper }}", "{}", "3"),
    ("guard_on_missing_role", "{% if messages[0].role == 'system' %}Y{% else %}N{% endif %}", "{\"messages\": [{\"content\": \"x\"}]}", "N"),
    ("range_global", "{% for i in range(3) %}{{ i }}{% endfor %}|{{ range(1, 6, 2) | list | join(',') }}", "{}", "012|1,3,5"),
    ("undefined_through_filters", "[{{ u | items | list }}][{{ u | list }}][{{ u | first }}][{{ u | string }}][{{ u | selectattr('a') | list }}][{{ u | join(',') }}][{{ u | lower }}][{{ u | map(attribute='a') | list }}]", "{}", "[[]][[]][][][[]][][][[]]"),
    ("in_with_undefined", "{{ 'a' in u }}|{{ 'a' not in u }}", "{}", "False|True"),
    ("dict_items_are_tuples", "{{ d | items | list }}", "{\"d\": {\"a\": 1}}", "[('a', 1)]"),
    ("dict_methods_repr", "{{ d.items() | list }}|{{ d.values() | list }}", "{\"d\": {\"a\": 1, \"b\": 2}}", "[('a', 1), ('b', 2)]|[1, 2]"),
    ("one_element_tuple_repr", "{{ (1,) }}", "{}", "(1,)"),
    ("adjacent_string_literals", "{{ 'a' 'b' }}", "{}", "ab"),
    ("set_block_then_filter", "{% set x %} y {% endset %}[{{ x | trim }}]", "{}", "[y]"),
    ("default_without_argument", "[{{ u | default }}]", "{}", "[]"),
    ("default_keeps_empty_mapping", "{{ {} | default('d') }}", "{}", "{}"),
    ("string_indexing", "{{ 'abc'[1] }}", "{}", "b"),
    ("inline_if_no_else_false", "[{{ 1 if false }}]", "{}", "[]"),
    ("string_filter_on_scalars", "{{ none | string }}|{{ true | string }}|{{ 1.0 | string }}", "{}", "None|True|1.0"),
    ("int_filter_conversions", "{{ 1.0 | int }}|{{ '1e3' | int }}|{{ ' 12 ' | int }}|{{ true | int }}", "{}", "1|1000|12|1"),
    ("split_returns_a_list", "{{ 'a-b'.split('-') }}", "{}", "['a', 'b']"),
    ("first_last_of_string", "{{ s | first }}|{{ s | last }}", "{\"s\": \"abc\"}", "a|c"),
    ("nested_macro_calls", "{% macro a(x) %}[{{ b(x) }}]{% endmacro %}{% macro b(x) %}<{{ x }}>{% endmacro %}{{ a('v') }}", "{}", "[<v>]"),
    ("macro_called_in_loop", "{% macro m(x) %}{{ x }}{% endmacro %}{% for i in xs %}{{ m(i) }}{% endfor %}", "{\"xs\": [1, 2]}", "12"),
    ("loop_inside_macro", "{% macro m(xs) %}{% for x in xs %}{{ loop.index }}{{ x }}{% endfor %}{% endmacro %}{{ m(['a','b']) }}", "{}", "1a2b"),
    ("macro_cannot_see_caller_locals", "{% macro m() %}[{{ localvar }}]{% endmacro %}{% for localvar in xs %}{{ m() }}{% endfor %}", "{\"xs\": [\"a\"]}", "[]"),
    ("selectattr_then_map", "{{ l | selectattr('r', 'equalto', 'u') | map(attribute='n') | join(',') }}", "{\"l\": [{\"n\": \"a\", \"r\": \"u\"}, {\"n\": \"b\", \"r\": \"t\"}]}", "a"),
    ("concat_with_none", "{{ none ~ 'x' }}", "{}", "Nonex"),
    ("nested_attribute_chain", "{{ a.b.c }}", "{\"a\": {\"b\": {\"c\": 1}}}", "1"),
    ("plus_marker_disables_lstrip", "a   {%+ if true %}b{% endif %}", "{}", "a   b"),
    ("comment_whitespace_markers", "a  {#- c -#}  b", "{}", "ab"),
    ("elif_chain_without_else", "[{% if v == 1 %}1{% elif v == 2 %}2{% endif %}]", "{\"v\": 9}", "[]"),
    ("empty_template", "", "{}", ""),
    ("only_a_comment", "[{# x #}]", "{}", "[]"),
    ("for_tuple_unpacking", "{% for a, b in pairs %}{{ a }}{{ b }};{% endfor %}", "{\"pairs\": [[1, 2], [3, 4]]}", "12;34;"),
    ("dict_get_with_default", "{{ d.get('z', 'D') }}", "{\"d\": {}}", "D"),
    ("startswith_tuple_argument", "{{ s.startswith(('x', 'ab')) }}", "{\"s\": \"abc\"}", "True"),
];

pub const EXTRA_TEMPLATE_CASES: &[(&str, &str, &str)] = &[
    ("tool_calls", "{\"messages\": [{\"role\": \"user\", \"content\": \"weather in Kyoto?\"}, {\"role\": \"assistant\", \"content\": \"\", \"tool_calls\": [{\"type\": \"function\", \"function\": {\"name\": \"get_weather\", \"arguments\": {\"city\": \"Kyoto\", \"days\": 3, \"verbose\": true}}}]}, {\"role\": \"tool\", \"content\": \"sunny\"}, {\"role\": \"user\", \"content\": \"thanks\"}], \"add_generation_prompt\": true}", "<|im_start|>system\nReasoning effort is set to xhigh. Please think carefully through the task, validate key assumptions, consider plausible alternatives, and prioritize correctness, consistency, and clarity in the final answer.<|im_end|>\n<|im_start|>user\nweather in Kyoto?<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n<tool_call>\n<function=get_weather>\n<parameter=city>\nKyoto\n</parameter>\n<parameter=days>\n3\n</parameter>\n<parameter=verbose>\ntrue\n</parameter>\n</function>\n</tool_call><|im_end|>\n<|im_start|>user\n<tool_response>\nsunny\n</tool_response><|im_end|>\n<|im_start|>user\nthanks<|im_end|>\n<|im_start|>assistant\n<think>\n"),
    ("tool_calls_with_text", "{\"messages\": [{\"role\": \"user\", \"content\": \"q\"}, {\"role\": \"assistant\", \"content\": \"let me look\", \"reasoning_content\": \"think\", \"tool_calls\": [{\"function\": {\"name\": \"f\", \"arguments\": {\"a\": \"b\"}}}, {\"function\": {\"name\": \"g\", \"arguments\": {}}}]}, {\"role\": \"tool\", \"content\": \"r1\"}, {\"role\": \"tool\", \"content\": \"r2\"}, {\"role\": \"user\", \"content\": \"ok\"}], \"add_generation_prompt\": true}", "<|im_start|>system\nReasoning effort is set to xhigh. Please think carefully through the task, validate key assumptions, consider plausible alternatives, and prioritize correctness, consistency, and clarity in the final answer.<|im_end|>\n<|im_start|>user\nq<|im_end|>\n<|im_start|>assistant\n<think>\nthink\n</think>\n\nlet me look\n\n<tool_call>\n<function=f>\n<parameter=a>\nb\n</parameter>\n</function>\n</tool_call>\n<tool_call>\n<function=g>\n</function>\n</tool_call><|im_end|>\n<|im_start|>user\n<tool_response>\nr1\n</tool_response>\n<tool_response>\nr2\n</tool_response><|im_end|>\n<|im_start|>user\nok<|im_end|>\n<|im_start|>assistant\n<think>\n"),
    ("vision_content", "{\"messages\": [{\"role\": \"user\", \"content\": [{\"type\": \"image\"}, {\"type\": \"text\", \"text\": \"what is this?\"}, {\"type\": \"video\"}]}], \"add_vision_id\": true, \"add_generation_prompt\": true}", "<|im_start|>system\nReasoning effort is set to xhigh. Please think carefully through the task, validate key assumptions, consider plausible alternatives, and prioritize correctness, consistency, and clarity in the final answer.<|im_end|>\n<|im_start|>user\nPicture 1: <|vision_start|><|image_pad|><|vision_end|>what is this?Video 1: <|vision_start|><|video_pad|><|vision_end|><|im_end|>\n<|im_start|>assistant\n<think>\n"),
    ("vision_content_no_id", "{\"messages\": [{\"role\": \"user\", \"content\": [{\"image_url\": \"x\"}, {\"text\": \"hi\"}]}], \"add_generation_prompt\": true}", "<|im_start|>system\nReasoning effort is set to xhigh. Please think carefully through the task, validate key assumptions, consider plausible alternatives, and prioritize correctness, consistency, and clarity in the final answer.<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>hi<|im_end|>\n<|im_start|>assistant\n<think>\n"),
    ("preserve_thinking_false", "{\"messages\": [{\"role\": \"user\", \"content\": \"a\"}, {\"role\": \"assistant\", \"content\": \"b\", \"reasoning_content\": \"r\"}, {\"role\": \"user\", \"content\": \"c\"}, {\"role\": \"assistant\", \"content\": \"d\", \"reasoning_content\": \"s\"}], \"preserve_thinking\": false, \"add_generation_prompt\": true}", "<|im_start|>system\nReasoning effort is set to xhigh. Please think carefully through the task, validate key assumptions, consider plausible alternatives, and prioritize correctness, consistency, and clarity in the final answer.<|im_end|>\n<|im_start|>user\na<|im_end|>\n<|im_start|>assistant\nb<|im_end|>\n<|im_start|>user\nc<|im_end|>\n<|im_start|>assistant\n<think>\ns\n</think>\n\nd<|im_end|>\n<|im_start|>assistant\n<think>\n"),
    ("no_generation_prompt", "{\"messages\": [{\"role\": \"user\", \"content\": \"a\"}], \"add_generation_prompt\": false}", "<|im_start|>system\nReasoning effort is set to xhigh. Please think carefully through the task, validate key assumptions, consider plausible alternatives, and prioritize correctness, consistency, and clarity in the final answer.<|im_end|>\n<|im_start|>user\na<|im_end|>\n"),
    ("enable_thinking_false_with_system", "{\"messages\": [{\"role\": \"system\", \"content\": \"sys\"}, {\"role\": \"user\", \"content\": \"u\"}], \"enable_thinking\": false, \"add_generation_prompt\": true}", "<|im_start|>system\nsys<|im_end|>\n<|im_start|>user\nu<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"),
    ("reasoning_effort_medium", "{\"messages\": [{\"role\": \"user\", \"content\": \"u\"}], \"reasoning_effort\": \"medium\", \"add_generation_prompt\": true}", "<|im_start|>user\nu<|im_end|>\n<|im_start|>assistant\n<think>\n"),
    ("tools_with_system", "{\"messages\": [{\"role\": \"system\", \"content\": \"sys\"}, {\"role\": \"user\", \"content\": \"u\"}], \"tools\": [{\"type\": \"function\", \"function\": {\"name\": \"f\", \"parameters\": {\"type\": \"object\"}}}], \"add_generation_prompt\": true}", "<|im_start|>system\nReasoning effort is set to xhigh. Please think carefully through the task, validate key assumptions, consider plausible alternatives, and prioritize correctness, consistency, and clarity in the final answer.\n\n# Tools\n\nYou have access to the following functions:\n\n<tools>\n{\"type\": \"function\", \"function\": {\"name\": \"f\", \"parameters\": {\"type\": \"object\"}}}\n</tools>\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n</parameter>\n<parameter=example_parameter_2>\nThis is the value for the second parameter\nthat can span\nmultiple lines\n</parameter>\n</function>\n</tool_call>\n\n<IMPORTANT>\nReminder:\n- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags\n- Required parameters MUST be specified\n- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after\n- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls\n</IMPORTANT>\n\nsys<|im_end|>\n<|im_start|>user\nu<|im_end|>\n<|im_start|>assistant\n<think>\n"),
    ("system_content_as_parts", "{\"messages\": [{\"role\": \"system\", \"content\": [{\"type\": \"text\", \"text\": \" padded \"}]}, {\"role\": \"user\", \"content\": \"u\"}], \"add_generation_prompt\": true}", "<|im_start|>system\nReasoning effort is set to xhigh. Please think carefully through the task, validate key assumptions, consider plausible alternatives, and prioritize correctness, consistency, and clarity in the final answer.\n\npadded<|im_end|>\n<|im_start|>user\nu<|im_end|>\n<|im_start|>assistant\n<think>\n"),
];

pub const RAISE_CASES: &[(&str, &str, &str)] = &[
    ("tool_response_only_user", "{\"messages\": [{\"role\": \"user\", \"content\": \"<tool_response>\\nx\\n</tool_response>\"}], \"add_generation_prompt\": true}", "No user query found in messages."),
    ("empty_messages", "{\"messages\": [], \"add_generation_prompt\": true}", "No messages provided."),
    ("bad_reasoning_effort", "{\"messages\": [{\"role\": \"user\", \"content\": \"u\"}], \"reasoning_effort\": \"turbo\", \"add_generation_prompt\": true}", "Unexpected reasoning effort turbo. Supported types are xhigh (default), medium, and low."),
    ("bad_role", "{\"messages\": [{\"role\": \"sysadmin\", \"content\": \"u\"}], \"add_generation_prompt\": true}", "No user query found in messages."),
    ("system_not_first", "{\"messages\": [{\"role\": \"user\", \"content\": \"u\"}, {\"role\": \"system\", \"content\": \"s\"}], \"add_generation_prompt\": true}", "System message must be at the beginning."),
    ("image_in_system", "{\"messages\": [{\"role\": \"system\", \"content\": [{\"type\": \"image\"}]}, {\"role\": \"user\", \"content\": \"u\"}], \"add_generation_prompt\": true}", "System message cannot contain images."),
    ("video_in_system", "{\"messages\": [{\"role\": \"system\", \"content\": [{\"type\": \"video\"}]}, {\"role\": \"user\", \"content\": \"u\"}], \"add_generation_prompt\": true}", "System message cannot contain videos."),
    ("bad_content_item", "{\"messages\": [{\"role\": \"user\", \"content\": [{\"nope\": 1}]}], \"add_generation_prompt\": true}", "Unexpected item type in content."),
    ("bad_content_type", "{\"messages\": [{\"role\": \"user\", \"content\": 42}], \"add_generation_prompt\": true}", "Unexpected content type."),
];

// ── helpers ──────────────────────────────────────────────────────────────────

fn real_template() -> JinjaTemplate {
    JinjaTemplate::compile(BONSAI2_CHAT_TEMPLATE).expect("the real chat template must compile")
}

fn context(json: &str) -> Value {
    Value::from_json_str(json).unwrap_or_else(|e| panic!("bad context JSON: {e}"))
}

/// Assert byte identity, reporting the first divergence instead of dumping
/// two multi-kilobyte prompts.
fn assert_same(case: &str, got: &str, want: &str) {
    if got == want {
        return;
    }
    let offset = got
        .as_bytes()
        .iter()
        .zip(want.as_bytes())
        .position(|(a, b)| a != b)
        .unwrap_or_else(|| got.len().min(want.len()));
    let from = offset.saturating_sub(48);
    panic!(
        "case '{case}' diverges at byte {offset}\n  got  ...{:?}\n  want ...{:?}\n  (got {} bytes, want {} bytes)",
        &got[from..got.len().min(offset + 48)],
        &want[from..want.len().min(offset + 48)],
        got.len(),
        want.len()
    );
}

fn tight_options() -> JinjaOptions {
    JinjaOptions {
        max_parse_depth: 32,
        max_render_depth: 48,
        max_loop_iterations: 2_000,
        max_output_bytes: 64 * 1024,
        ..JinjaOptions::default()
    }
}

// ── the acceptance gate: the real template, byte for byte ────────────────────

#[test]
fn jinja_renders_the_real_chat_template_byte_identically() {
    let template = real_template();
    for (name, ctx_json, expected) in GOLDEN_CASES {
        let rendered = template
            .render(&context(ctx_json))
            .unwrap_or_else(|e| panic!("golden '{name}' failed to render: {e}"));
        assert_same(name, &rendered, expected);
    }
    assert_eq!(
        GOLDEN_CASES.len(),
        5,
        "all five fork goldens must be covered"
    );
}

#[test]
fn jinja_renders_the_real_template_tool_and_vision_branches() {
    let template = real_template();
    for (name, ctx_json, expected) in EXTRA_TEMPLATE_CASES {
        let rendered = template
            .render(&context(ctx_json))
            .unwrap_or_else(|e| panic!("case '{name}' failed to render: {e}"));
        assert_same(name, &rendered, expected);
    }
}

#[test]
fn jinja_raise_exception_surfaces_as_template_raise() {
    let template = real_template();
    for (name, ctx_json, message) in RAISE_CASES {
        match template.render(&context(ctx_json)) {
            Err(JinjaError::TemplateRaise(actual)) => assert_eq!(&actual, message, "case '{name}'"),
            other => panic!("case '{name}': expected TemplateRaise({message:?}), got {other:?}"),
        }
    }
    assert!(
        RAISE_CASES.len() >= 9,
        "every raise_exception path in the template must be covered"
    );

    // And the standalone global, including the error accessor the chat template uses.
    let err = JinjaTemplate::compile("{{ raise_exception('boom') }}")
        .expect("compile")
        .render(&Value::Undefined)
        .expect_err("must raise");
    assert_eq!(err.template_raise_message(), Some("boom"));
    assert_eq!(err.to_string(), "boom");
}

/// The context must be built from JSON **text**, not routed through
/// `serde_json::Value`: without that crate's `preserve_order` feature (it is
/// off in this workspace) its `Map` is a `BTreeMap`, so object keys come back
/// sorted and `tojson` then emits `{"description": …, "name": …}` where the
/// reference runtime emits `{"name": …, "description": …}`.  Golden case 5
/// would change byte for byte.  The order is destroyed at parse time, so it
/// cannot be recovered afterwards — this test pins the working path and makes
/// the trap executable.
#[test]
fn jinja_tool_key_order_requires_from_json_str() {
    const TOOL: &str = r#"{"name": "get_weather", "description": "Get weather"}"#;

    let template = JinjaTemplate::compile("{{ t | tojson }}").expect("compile");

    let mut ordered = ValueMap::new();
    ordered.insert("t", Value::from_json_str(TOOL).expect("ordered parse"));
    assert_eq!(
        template.render(&Value::map(ordered)).expect("render"),
        TOOL,
        "Value::from_json_str must preserve object key order"
    );

    let through_serde: serde_json::Value = serde_json::from_str(TOOL).expect("serde parse");
    let mut converted = ValueMap::new();
    converted.insert("t", Value::from(&through_serde));
    let rendered = template.render(&Value::map(converted)).expect("render");
    assert!(
        rendered == TOOL || rendered == r#"{"description": "Get weather", "name": "get_weather"}"#,
        "unexpected serde_json ordering: {rendered}"
    );
}

/// The template does `tool_call.arguments|items`, so `arguments` must already
/// be a mapping.  OpenAI's wire format delivers it as a JSON *string*; that
/// path errors (exactly as Python Jinja does) instead of rendering nonsense,
/// so a caller has to parse it into an object first.
#[test]
fn jinja_tool_arguments_must_be_a_mapping() {
    let template =
        JinjaTemplate::compile("{% for k, v in a | items %}{{ k }}{% endfor %}").expect("compile");
    assert!(matches!(
        template.render(&context(r#"{"a": "{\"city\": \"Kyoto\"}"}"#)),
        Err(JinjaError::Runtime(_))
    ));
    assert_eq!(
        template
            .render(&context(r#"{"a": {"city": "Kyoto"}}"#))
            .expect("render"),
        "city"
    );
}

// ── differential corpus against the Python oracle ────────────────────────────

#[test]
fn jinja_micro_cases_match_the_python_oracle() {
    for (name, source, ctx_json, expected) in MICRO_CASES {
        let template = JinjaTemplate::compile(source)
            .unwrap_or_else(|e| panic!("case '{name}' failed to compile: {e}\n  {source}"));
        let rendered = template
            .render(&context(ctx_json))
            .unwrap_or_else(|e| panic!("case '{name}' failed to render: {e}\n  {source}"));
        assert_same(name, &rendered, expected);
    }
    assert!(MICRO_CASES.len() >= 100);
}

#[test]
fn jinja_tojson_matches_python_json_dumps_defaults() {
    // The nested tools object from the golden set: separators ", " / ": ",
    // insertion order preserved (json.dumps does NOT sort keys), and no HTML
    // escaping of < > & ' (which Jinja's own `tojson` would apply).
    let tool = r#"{"type": "function", "function": {"name": "get_weather", "description": "Get weather", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}"#;
    let rendered = JinjaTemplate::compile("{{ t | tojson }}")
        .expect("compile")
        .render(&context(&format!("{{\"t\": {tool}}}")))
        .expect("render");
    assert_eq!(rendered, tool);

    let cases: &[(&str, &str)] = &[
        (
            r#"{"v": {"z": 1, "m": 2, "a": 3}}"#,
            r#"{"z": 1, "m": 2, "a": 3}"#,
        ),
        (
            r#"{"v": [1, 2.5, true, false, null, ""]}"#,
            r#"[1, 2.5, true, false, null, ""]"#,
        ),
        (r#"{"v": {}}"#, "{}"),
        (r#"{"v": {"a": 1.0, "b": 0.1}}"#, r#"{"a": 1.0, "b": 0.1}"#),
    ];
    let template = JinjaTemplate::compile("{{ v | tojson }}").expect("compile");
    for (ctx_json, expected) in cases {
        let rendered = template.render(&context(ctx_json)).expect("render");
        assert_eq!(&rendered, expected, "for {ctx_json}");
    }

    // ensure_ascii, the `[^ -~]` rule (DEL is escaped) and surrogate
    // pairs — built from chars so no source-level escape can hide a
    // mismatch, and deliberately including the `< > & '` that Jinja's own
    // `tojson` would html-escape but `json.dumps` does not.
    let mut specials = ValueMap::new();
    specials.insert(
        "v",
        Value::str("< > & ' \" \\ \n \t \u{7f} \u{e9} \u{1f600}"),
    );
    assert_eq!(
        template.render(&Value::map(specials)).expect("render"),
        "\"< > & ' \\\" \\\\ \\n \\t \\u007f \\u00e9 \\ud83d\\ude00\""
    );

    // Values that have no JSON form must error, not render something wrong.
    for source in [
        "{{ nope | tojson }}",
        "{% set n = namespace() %}{{ n | tojson }}",
    ] {
        assert!(matches!(
            JinjaTemplate::compile(source)
                .expect("compile")
                .render(&Value::Undefined),
            Err(JinjaError::Runtime(_))
        ));
    }
}

// ── the finding: malformed / unsupported input must ERROR ────────────────────

#[test]
fn jinja_malformed_templates_are_syntax_errors_with_a_position() {
    let malformed: &[&str] = &[
        "{{",
        "{{ a",
        "{{ a }",
        "{%",
        "{% if a",
        "{% if a %}",
        "{% if %}x{% endif %}",
        "{% for %}{% endfor %}",
        "{% for x %}{% endfor %}",
        "{% for x in %}{% endfor %}",
        "{% endif %}",
        "{% endfor %}",
        "{% else %}",
        "{% macro %}{% endmacro %}",
        "{% macro m( %}{% endmacro %}",
        "{% set %}",
        "{% set x = %}",
        "{% set x %}",
        "{# unterminated",
        "{{ 'unterminated }}",
        "{{ \"unterminated }}",
        "{{ a. }}",
        "{{ a[ }}",
        "{{ a[] }}",
        "{{ (1, }}",
        "{{ {1:} }}",
        "{{ @ }}",
        "{{ a ~ }}",
        "{{ | trim }}",
        "{{ a | }}",
        "{{ a is }}",
        "{{ a is not }}",
        "{{ f(a=) }}",
        "{{ f(*a) }}",
        "{{ a[1,2] }}",
        "{{ a[1:2:3:4] }}",
        "{% break %}",
        "{% continue %}",
        "{% call %}{% endcall %}",
        "{% call notacall %}{% endcall %}",
        "{% if a %}x{% endfor %}",
        "{% for x in y %}x{% endif %}",
        "{% macro m() %}x{% endif %}",
        "{% generation %}x",
    ];
    for source in malformed {
        match JinjaTemplate::compile(source) {
            Err(JinjaError::Syntax { line, column, .. }) => {
                assert!(line >= 1, "{source:?} reported line {line}");
                assert!(column >= 1, "{source:?} reported column {column}");
            }
            other => panic!("{source:?} must be a syntax error, got {other:?}"),
        }
    }
}

#[test]
fn jinja_unsupported_constructs_are_rejected_not_ignored() {
    let unsupported: &[&str] = &[
        "{% include 'other.jinja' %}",
        "{% extends 'base.jinja' %}",
        "{% block body %}{% endblock %}",
        "{% raw %}{{ literal }}{% endraw %}",
        "{% filter upper %}x{% endfilter %}",
        "{% with a = 1 %}{% endwith %}",
        "{% import 'x' as y %}",
        "{% from 'x' import y %}",
        "{% do a.append(1) %}",
        "{% autoescape true %}{% endautoescape %}",
        "{% trans %}x{% endtrans %}",
        "{% nosuchstatement %}",
        "{{ a | nosuchfilter }}",
        "{{ a | Tojson }}",
        "{{ a is nosuchtest }}",
        "{{ a is Defined }}",
        "{% for x in y recursive %}{% endfor %}",
    ];
    for source in unsupported {
        assert!(
            matches!(
                JinjaTemplate::compile(source),
                Err(JinjaError::Syntax { .. })
            ),
            "{source:?} must be rejected at compile time"
        );
    }
}

#[test]
fn jinja_runtime_type_errors_are_reported_not_papered_over() {
    let cases: &[(&str, &str)] = &[
        ("{{ 'a' + 1 }}", "{}"),
        ("{{ 1 / 0 }}", "{}"),
        ("{{ 1 // 0 }}", "{}"),
        ("{{ 1 % 0 }}", "{}"),
        ("{{ 3 | length }}", "{}"),
        // Jinja's `int` filter raises on an undefined value instead of
        // silently returning its default — matched here on purpose.
        ("{{ nope | int }}", "{}"),
        ("{{ n | length }}", r#"{"n": null}"#),
        ("{{ -'a' }}", "{}"),
        ("{{ 1 in 3 }}", "{}"),
        ("{{ 1 < 'a' }}", "{}"),
        ("{% for x in n %}{{ x }}{% endfor %}", r#"{"n": null}"#),
        ("{% for x in i %}{{ x }}{% endfor %}", r#"{"i": 3}"#),
        ("{{ messages[0].role }}", r#"{"messages": []}"#),
        ("{{ d.a.b }}", r#"{"d": {}}"#),
        ("{{ nope.attr }}", "{}"),
        ("{{ s.nosuchmethod() }}", r#"{"s": "x"}"#),
        ("{{ nosuchfunction('x') }}", "{}"),
        ("{% set d = 1 %}{% set d.x = 2 %}", "{}"),
        (
            "{% for a, b in xs %}{{ a }}{% endfor %}",
            r#"{"xs": [[1, 2, 3]]}"#,
        ),
        ("{{ xs[::0] }}", r#"{"xs": [1]}"#),
    ];
    for (source, ctx_json) in cases {
        let template = JinjaTemplate::compile(source)
            .unwrap_or_else(|e| panic!("{source:?} should compile, got {e}"));
        match template.render(&context(ctx_json)) {
            Err(JinjaError::Runtime(_)) => {}
            other => panic!("{source:?} must be a runtime error, got {other:?}"),
        }
    }
}

#[test]
fn jinja_undefined_is_soft_except_for_attribute_access() {
    let soft: &[(&str, &str)] = &[
        ("[{{ nope }}]", "[]"),
        ("[{{ nope | trim }}]", "[]"),
        ("[{{ nope | length }}]", "[0]"),
        ("[{{ nope | default('d') }}]", "[d]"),
        ("[{% if nope %}T{% else %}F{% endif %}]", "[F]"),
        ("[{{ nope is undefined }}]", "[True]"),
        ("[{% for x in nope %}{{ x }}{% endfor %}]", "[]"),
        ("[{{ d['missing'] }}]", "[]"),
        ("[{{ xs[99] }}]", "[]"),
    ];
    for (source, expected) in soft {
        let rendered = JinjaTemplate::compile(source)
            .expect("compile")
            .render(&context(r#"{"d": {}, "xs": [1]}"#))
            .unwrap_or_else(|e| panic!("{source:?} should render, got {e}"));
        assert_eq!(&rendered, expected, "for {source}");
    }
}

// ── hostile input: bounded, never hanging ────────────────────────────────────

#[test]
fn jinja_hostile_templates_are_bounded() {
    let options = tight_options();

    // A loop bomb over attacker-supplied data.
    let bomb =
        "{% for a in xs %}{% for b in xs %}{% for c in xs %}x{% endfor %}{% endfor %}{% endfor %}";
    let items: Vec<String> = (0..100).map(|i| i.to_string()).collect();
    let ctx = context(&format!("{{\"xs\": [{}]}}", items.join(",")));
    let rendered = JinjaTemplate::compile_with(bomb, options.clone())
        .expect("compile")
        .render(&ctx);
    assert!(
        matches!(rendered, Err(JinjaError::Limit(_))),
        "{rendered:?}"
    );

    // Unbounded output growth.
    let grow = "{% for i in xs %}0123456789abcdef{% endfor %}";
    let narrow = JinjaOptions {
        max_output_bytes: 512,
        ..tight_options()
    };
    let rendered = JinjaTemplate::compile_with(grow, narrow)
        .expect("compile")
        .render(&ctx);
    assert!(
        matches!(rendered, Err(JinjaError::Limit(_))),
        "{rendered:?}"
    );

    // Deep syntactic nesting must not blow the stack.
    let deep_expr = format!("{{{{ {}1{} }}}}", "(".repeat(4096), ")".repeat(4096));
    assert!(matches!(
        JinjaTemplate::compile_with(&deep_expr, options.clone()),
        Err(JinjaError::Syntax { .. })
    ));
    let deep_blocks = format!(
        "{}x{}",
        "{% if a %}".repeat(4096),
        "{% endif %}".repeat(4096)
    );
    assert!(matches!(
        JinjaTemplate::compile_with(&deep_blocks, options.clone()),
        Err(JinjaError::Syntax { .. })
    ));
    let deep_lists = format!("{{{{ {}{} }}}}", "[".repeat(4096), "]".repeat(4096));
    assert!(matches!(
        JinjaTemplate::compile_with(&deep_lists, options.clone()),
        Err(JinjaError::Syntax { .. })
    ));

    // Multiplication cannot be used to allocate without bound.
    assert!(
        JinjaTemplate::compile_with("{{ 'x' * 99999999 }}", options.clone())
            .expect("compile")
            .render(&Value::Undefined)
            .is_err()
    );

    // A self-referential namespace must not recurse forever.
    let cyclic = "{% set ns = namespace() %}{% set ns.me = ns %}{{ ns }}";
    assert!(JinjaTemplate::compile_with(cyclic, options)
        .expect("compile")
        .render(&Value::Undefined)
        .is_err());
}

#[test]
fn jinja_limits_are_configurable_and_enforced() {
    let options = JinjaOptions {
        max_loop_iterations: 4,
        ..JinjaOptions::default()
    };
    let template =
        JinjaTemplate::compile_with("{% for i in range(10) %}{{ i }}{% endfor %}", options)
            .expect("compile");
    assert!(matches!(
        template.render(&Value::Undefined),
        Err(JinjaError::Limit(_))
    ));
}

// ── fuzzing: the lexer and parser must be total ──────────────────────────────

fn fragments() -> Vec<&'static str> {
    vec![
        "{{",
        "}}",
        "{%",
        "%}",
        "{#",
        "#}",
        "{{-",
        "-}}",
        "{%-",
        "-%}",
        "{",
        "}",
        "|",
        "~",
        ".",
        "[",
        "]",
        "(",
        ")",
        ",",
        ":",
        "=",
        "==",
        "!=",
        "<",
        ">",
        "+",
        "-",
        "*",
        "/",
        "//",
        "%",
        "**",
        "'",
        "\"",
        "'a'",
        "\"b\"",
        "\\",
        "if",
        "elif",
        "else",
        "endif",
        "for",
        "in",
        "endfor",
        "set",
        "endset",
        "macro",
        "endmacro",
        "call",
        "endcall",
        "generation",
        "endgeneration",
        "break",
        "continue",
        "is",
        "not",
        "and",
        "or",
        "none",
        "true",
        "false",
        "messages",
        "loop",
        "ns",
        "tojson",
        "trim",
        "length",
        "default",
        "items",
        "selectattr",
        "namespace",
        "raise_exception",
        "0",
        "1",
        "42",
        "1.5",
        "1e9",
        "_",
        "x",
        " ",
        "\n",
        "\t",
        "\r\n",
        "é",
        "😀",
        "\u{7f}",
        "<|im_start|>",
    ]
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(400))]

    /// Structured fuzzing: template-shaped noise must compile or error, never
    /// panic and never loop forever.
    #[test]
    fn jinja_fuzz_structured_source_never_panics(
        parts in proptest::collection::vec(proptest::sample::select(fragments()), 0..40)
    ) {
        let source: String = parts.concat();
        let _ = JinjaTemplate::compile_with(&source, tight_options());
    }

    /// Unstructured fuzzing over arbitrary Unicode text.
    #[test]
    fn jinja_fuzz_arbitrary_text_never_panics(source in "\\PC{0,160}") {
        let _ = JinjaTemplate::compile_with(&source, tight_options());
    }

    /// Anything that survives compilation must also render without panicking
    /// (and within the configured bounds).
    #[test]
    fn jinja_fuzz_rendering_never_panics(
        parts in proptest::collection::vec(proptest::sample::select(fragments()), 0..40)
    ) {
        let source: String = parts.concat();
        if let Ok(template) = JinjaTemplate::compile_with(&source, tight_options()) {
            let ctx = context(r#"{"messages": [{"role": "user", "content": "hi"}], "xs": [1, 2], "d": {"a": 1}, "s": "abc"}"#);
            let _ = template.render(&ctx);
        }
    }

    /// The serve-path threat is attacker-influenced **data**, not an
    /// attacker-supplied template: throw arbitrary contexts at the real chat
    /// template, which exercises the macro, the reverse slice, `tojson` and
    /// every `raise_exception` branch.  Any outcome is acceptable except a
    /// panic or a hang.
    #[test]
    fn jinja_fuzz_hostile_context_never_panics(
        roles in proptest::collection::vec(
            proptest::sample::select(vec!["user", "assistant", "system", "tool", "\u{1f600}", ""]),
            0..6,
        ),
        contents in proptest::collection::vec(
            proptest::sample::select(vec![
                "\"hi\"", "\"\"", "null", "42", "[]", "[{\"text\": \"t\"}]",
                "[{\"type\": \"image\"}]", "[{\"nope\": 1}]", "{\"k\": \"v\"}",
                "\"<tool_response>\\nx\\n</tool_response>\"", "\"<|im_start|>\"",
            ]),
            0..6,
        ),
        effort in proptest::sample::select(vec!["\"xhigh\"", "\"low\"", "\"medium\"", "\"nope\"", "null", "7"]),
        thinking in proptest::sample::select(vec!["true", "false", "null", "\"yes\""]),
    ) {
        let messages: Vec<String> = roles
            .iter()
            .zip(contents.iter().chain(std::iter::repeat(&"null")))
            .map(|(role, content)| format!("{{\"role\": \"{role}\", \"content\": {content}}}"))
            .collect();
        let ctx_json = format!(
            "{{\"messages\": [{}], \"reasoning_effort\": {effort}, \"enable_thinking\": {thinking}, \"add_generation_prompt\": true}}",
            messages.join(", ")
        );
        let template = JinjaTemplate::compile_with(BONSAI2_CHAT_TEMPLATE, tight_options())
            .expect("the real template compiles");
        if let Ok(ctx) = Value::from_json_str(&ctx_json) {
            let _ = template.render(&ctx);
        }
    }
}
