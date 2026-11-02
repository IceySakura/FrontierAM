# Universe 题目 API / 凭据依赖矩阵

由 `uv run python -m nexus.extensions.tasks.mcp_benchmark.universe_selection matrix` 生成。

- `依赖 MCP 服务`: 这道题会加载的 MCP 服务集合。
- `需要额外凭据的服务`: 在 `original` 模式下需要额外准备凭据或第三方账号配置的服务。
- `api-free 可跑`: 在缺少 original 额外凭据时，当前 benchmark 是否还能以 best-effort 方式运行这道题。

| case_id | 题目摘要 | 依赖 MCP 服务 | 需要额外凭据的服务 | 凭据字段名 | api-free 可跑 | api-free 说明 |
| --- | --- | --- | --- | --- | --- | --- |
| blender_task_0001 | Clear the current Blender scene and remove all PolyHaven materials. Create a... | blender | - | - | 是 | - |
| blender_task_0002 | Clear the current Blender scene and remove all PolyHaven materials. Create a... | blender | - | - | 是 | - |
| blender_task_0003 | Clear the current Blender scene and remove all PolyHaven materials. Create a... | blender | - | - | 是 | - |
| blender_task_0004 | Clear the current Blender scene and remove all PolyHaven materials. Add a Suz... | blender | - | - | 是 | - |
| blender_task_0005 | Clear the current Blender scene and remove all PolyHaven materials. Create a... | blender | - | - | 是 | - |
| blender_task_0006 | Clear the current Blender scene and remove all PolyHaven materials. Create a... | blender | - | - | 是 | - |
| blender_task_0007 | Clear the current Blender scene and remove all PolyHaven materials. Create a... | blender | - | - | 是 | - |
| blender_task_0008 | Clear the current Blender scene and remove all PolyHaven materials. Create a... | blender | - | - | 是 | - |
| blender_task_0009 | Clear the current Blender scene and remove all PolyHaven materials. Create a... | blender | - | - | 是 | - |
| blender_task_0010 | Clear the current Blender scene and remove all PolyHaven materials. Search Po... | blender | - | - | 是 | - |
| blender_task_0011 | Clear the current Blender scene and remove all PolyHaven materials. Set the r... | blender | - | - | 是 | - |
| blender_task_0012 | Clear the current Blender scene and remove all PolyHaven materials. Set the r... | blender | - | - | 是 | - |
| blender_task_0013 | Clear the current Blender scene and remove all PolyHaven materials. Set the o... | blender | - | - | 是 | - |
| blender_task_0014 | Clear the current Blender scene and remove all PolyHaven materials. Remove an... | blender | - | - | 是 | - |
| blender_task_0015 | Clear the current Blender scene and remove all PolyHaven materials. Ensure th... | blender | - | - | 是 | - |
| blender_task_0016 | Clear the current Blender scene and remove all PolyHaven materials. Create a... | blender | - | - | 是 | - |
| blender_task_0017 | Clear the current Blender scene and remove all PolyHaven materials. Ensure th... | blender | - | - | 是 | - |
| blender_task_0019 | Clear the current Blender scene and remove all PolyHaven materials. Create a... | blender | - | - | 是 | - |
| blender_task_0020 | Clear the current Blender scene and remove all PolyHaven materials. Create th... | blender | - | - | 是 | - |
| github_task_0001 | Hey! I need to set up a new repo for this travel planner project I'm working... | github | github | github_token | 否 | api-free 模式当前仍会 fallback 到 live GitHub，因此仍需要 github_token |
| github_task_0002 | Hi! I'm just starting to learn about LLMs and want to set up my first trainin... | github | github | github_token | 否 | api-free 模式当前仍会 fallback 到 live GitHub，因此仍需要 github_token |
| github_task_0003 | I'm working on a comprehensive evaluation framework for video-based language... | github | github | github_token | 否 | api-free 模式当前仍会 fallback 到 live GitHub，因此仍需要 github_token |
| github_task_0004 | For this assignment, I would like you to establish a new project repository n... | github | github | github_token | 否 | api-free 模式当前仍会 fallback 到 live GitHub，因此仍需要 github_token |
| github_task_0005 | I am a first-year PhD student in Computer Science. My supervisor has assigned... | github | github | github_token | 否 | api-free 模式当前仍会 fallback 到 live GitHub，因此仍需要 github_token |
| github_task_0006 | As a Facebook ecosystem analyzer, I need you to conduct a comprehensive analy... | github | github | github_token | 否 | api-free 模式当前仍会 fallback 到 live GitHub，因此仍需要 github_token |
| github_task_0007 | Hi! I'm learning how to use GitHub and I want to practice exploring repositor... | github | github | github_token | 否 | api-free 模式当前仍会 fallback 到 live GitHub，因此仍需要 github_token |
| github_task_0008 | As an open-source enthusiast and agent developer, I'm deeply inspired by the... | github | github | github_token | 否 | api-free 模式当前仍会 fallback 到 live GitHub，因此仍需要 github_token |
| github_task_0009 | Hey there! I'm working on a project to analyze Microsoft's repositories and n... | github | github | github_token | 否 | api-free 模式当前仍会 fallback 到 live GitHub，因此仍需要 github_token |
| github_task_0010 | I need to analyze Microsoft's Air-related repositories for open issues that l... | github | github | github_token | 否 | api-free 模式当前仍会 fallback 到 live GitHub，因此仍需要 github_token |
| github_task_0011 | Hi there! I'm a master's student working on AI research and I've come across... | github | github | github_token | 否 | api-free 模式当前仍会 fallback 到 live GitHub，因此仍需要 github_token |
| github_task_0012 | There are two repositories: QwenLM's Qwen2.5-VL and deepseek-ai's DeepSeek-VL... | github | github | github_token | 否 | api-free 模式当前仍会 fallback 到 live GitHub，因此仍需要 github_token |
| github_task_0014 | Hey! So I've got these 4 repositories to work with: QwenLM's Qwen2.5-VL, deep... | github | github | github_token | 否 | api-free 模式当前仍会 fallback 到 live GitHub，因此仍需要 github_token |
| github_task_0015 | Oh wow, I'm absolutely fascinated by these amazing VLM repositories! I've bee... | github | github | github_token | 否 | api-free 模式当前仍会 fallback 到 live GitHub，因此仍需要 github_token |
| github_task_0016 | Hi! I'm a student working on my final project and I really need help setting... | github | github | github_token | 否 | api-free 模式当前仍会 fallback 到 live GitHub，因此仍需要 github_token |
| github_task_0017 | Hi! I'm a student working on learning GitHub automation and I really need you... | github | github | github_token | 否 | api-free 模式当前仍会 fallback 到 live GitHub，因此仍需要 github_token |
| github_task_0018 | Hi! I'm a student working on learning GitHub automation and I really need you... | github | github | github_token | 否 | api-free 模式当前仍会 fallback 到 live GitHub，因此仍需要 github_token |
| github_task_0019 | Hi there! I'm working on a GitHub automation project and could really use you... | github | github | github_token | 否 | api-free 模式当前仍会 fallback 到 live GitHub，因此仍需要 github_token |
| github_task_0021 | Hi! I'm Kai, a PhD student in Computer Science. For my research project, I ne... | github | github | github_token | 否 | api-free 模式当前仍会 fallback 到 live GitHub，因此仍需要 github_token |
| github_task_0022 | Hi there! I'm leading a research initiative on agent evaluation frameworks. I... | github | github | github_token | 否 | api-free 模式当前仍会 fallback 到 live GitHub，因此仍需要 github_token |
| github_task_0023 | Hi! I need help with a research project. Could you please search for reposito... | github | github | github_token | 否 | api-free 模式当前仍会 fallback 到 live GitHub，因此仍需要 github_token |
| github_task_0024 | Hi! I'm absolutely thrilled to be working on a research project involving BLI... | github | github | github_token | 否 | api-free 模式当前仍会 fallback 到 live GitHub，因此仍需要 github_token |
| github_task_0025 | There are two repositories: Salesforce's LAVIS and Salesforce's BLIP. Fork th... | github | github | github_token | 否 | api-free 模式当前仍会 fallback 到 live GitHub，因此仍需要 github_token |
| github_task_0026 | Hi! I'm absolutely thrilled to be working with two of the most incredible and... | github | github | github_token | 否 | api-free 模式当前仍会 fallback 到 live GitHub，因此仍需要 github_token |
| github_task_0027 | Hi! Can you please help me make a super cool project repository called `comme... | github | github | github_token | 否 | api-free 模式当前仍会 fallback 到 live GitHub，因此仍需要 github_token |
| github_task_0028 | Hi! Can you please help me make a super cool project repository called `comme... | github | github | github_token | 否 | api-free 模式当前仍会 fallback 到 live GitHub，因此仍需要 github_token |
| github_task_0029 | Hello! I'm Alex, a software engineer working on an exciting new project. I ne... | github | github | github_token | 否 | api-free 模式当前仍会 fallback 到 live GitHub，因此仍需要 github_token |
| github_task_0030 | Hi there! I'm leading a research initiative on agent evaluation frameworks an... | github | github | github_token | 否 | api-free 模式当前仍会 fallback 到 live GitHub，因此仍需要 github_token |
| google_maps_task_0001 | Hey! I'm so excited to plan an amazing road trip starting in 'Johor Bahru' an... | google-maps | google-maps | google_cloud_console_api_key | 是 | - |
| google_maps_task_0002 | Hello! We just got married and we're planning our dream honeymoon road trip f... | google-maps | google-maps | google_cloud_console_api_key | 是 | - |
| google_maps_task_0003 | I'm excited to plan an amazing road trip starting in 'San Francisco' and endi... | google-maps | google-maps | google_cloud_console_api_key | 是 | - |
| google_maps_task_0004 | Hi! My partner and I are planning a special pre-wedding road trip from 'Los A... | google-maps | google-maps | google_cloud_console_api_key | 是 | - |
| google_maps_task_0005 | My dear wife and I are planning what might be our last grand adventure togeth... | google-maps | google-maps | google_cloud_console_api_key | 是 | - |
| google_maps_task_0006 | Hey! My friend and I are planning an epic road trip adventure from 'Seattle'... | google-maps | google-maps | google_cloud_console_api_key | 是 | - |
| google_maps_task_0007 | Konnichiwa! I'm a girl from Japan and I'm so excited about my dream road trip... | google-maps | google-maps | google_cloud_console_api_key | 是 | - |
| google_maps_task_0008 | Heyy! I'm a young girl from Africa and I'm super excited about my upcoming ro... | google-maps | google-maps | google_cloud_console_api_key | 是 | - |
| google_maps_task_0009 | My wife and I are planning a wonderful road trip from 'Sydney' to 'Brisbane',... | google-maps | google-maps | google_cloud_console_api_key | 是 | - |
| google_maps_task_0010 | Hey there! Me and my college buddies are planning this epic road trip from 'O... | google-maps | google-maps | google_cloud_console_api_key | 是 | - |
| google_maps_task_0011 | I need to drive from the Merlion Park, Singapore to the Petronas Towers, Kual... | google-maps | google-maps | google_cloud_console_api_key | 是 | - |
| google_maps_task_0012 | I have a business trip that requires driving from Merlion Park, Singapore to... | google-maps | google-maps | google_cloud_console_api_key | 是 | - |
| google_maps_task_0013 | I'm organizing a romantic getaway with my partner, starting at the renowned S... | google-maps | google-maps | google_cloud_console_api_key | 是 | - |
| google_maps_task_0014 | My spouse and I are newlyweds planning our first adventure together as a marr... | google-maps | google-maps | google_cloud_console_api_key | 是 | - |
| google_maps_task_0015 | I'm organizing a romantic getaway with my partner, starting at the iconic Eif... | google-maps | google-maps | google_cloud_console_api_key | 是 | - |
| google_maps_task_0016 | We're planning a romantic getaway together, starting at the iconic Eiffel Tow... | google-maps | google-maps | google_cloud_console_api_key | 是 | - |
| google_maps_task_0017 | We're planning an amazing family adventure from Disneyland in Anaheim to Yose... | google-maps | google-maps | google_cloud_console_api_key | 是 | - |
| google_maps_task_0018 | My wife and I are planning an amazing family adventure from Disneyland in Ana... | google-maps | google-maps | google_cloud_console_api_key | 是 | - |
| google_maps_task_0019 | As a mom, I'm planning the most wonderful family adventure from the Grand Can... | google-maps | google-maps | google_cloud_console_api_key | 是 | - |
| google_maps_task_0020 | We're planning the most amazing family adventure from the breathtaking Grand... | google-maps | google-maps | google_cloud_console_api_key | 是 | - |
| google_maps_task_0021 | I live in Kent Ridge Hill Residences, Singapore. One of my friends lives in S... | google-maps | google-maps | google_cloud_console_api_key | 是 | - |
| google_maps_task_0022 | I live in Kent Ridge Hill Residences, Singapore. My friend lives in Symphony... | google-maps | google-maps | google_cloud_console_api_key | 是 | - |
| google_maps_task_0023 | I am based at the Empire State Building in New York City. My business partner... | google-maps | google-maps | google_cloud_console_api_key | 是 | - |
| google_maps_task_0024 | I live in the Empire State Building in New York City. My colleague lives in t... | google-maps | google-maps | google_cloud_console_api_key | 是 | - |
| google_maps_task_0025 | I live in the Sheraton Jumeirah Beach Resort in Dubai. My colleague lives in... | google-maps | google-maps | google_cloud_console_api_key | 是 | - |
| google_maps_task_0026 | I'm currently staying at the Sheraton Jumeirah Beach Resort in Dubai. My firs... | google-maps | google-maps | google_cloud_console_api_key | 是 | - |
| google_maps_task_0027 | I live in J Gateway Condo, Singapore. My friend lives in Park Place Residence... | google-maps | google-maps | google_cloud_console_api_key | 是 | - |
| google_maps_task_0028 | I live in J Gateway Condo, Singapore. My friend lives in Park Place Residence... | google-maps | google-maps | google_cloud_console_api_key | 是 | - |
| google_maps_task_0029 | I live in Kent Ridge Hill Residences, Singapore. My one friend lives in Symph... | google-maps | google-maps | google_cloud_console_api_key | 是 | - |
| google_maps_task_0030 | Hey girls! So I'm staying at J Gateway Condo in Singapore, and I really want... | google-maps | google-maps | google_cloud_console_api_key | 是 | - |
| google_maps_task_0031 | Identify 1 'swimming pool' location in Singapore that are south of the latitu... | google-maps | google-maps | google_cloud_console_api_key | 是 | - |
| google_maps_task_0032 | Identify 1 'library' location in New York City that are north of the latitude... | google-maps | google-maps | google_cloud_console_api_key | 是 | - |
| google_maps_task_0033 | Identify 1 'stadium' location in UK that are south of the latitude of 'London... | google-maps | google-maps | google_cloud_console_api_key | 是 | - |
| google_maps_task_0034 | Identify 1 'hospital' location in Pairs, France that are north of the latitud... | google-maps | google-maps | google_cloud_console_api_key | 是 | - |
| google_maps_task_0035 | Identify 1 'museum' location in Rome, Italy that are south of the latitude of... | google-maps | google-maps | google_cloud_console_api_key | 是 | - |
| info_search_task_0001 | I'm looking for someone based on the clues below: - Score 16 goals in 2024-25... | google-search<br>fetch | google-search | serper_api_key | 是 | - |
| info_search_task_0002 | I'm looking for someone based on the clues below: - Played for the SAC (NBA)... | google-search<br>fetch | google-search | serper_api_key | 是 | - |
| info_search_task_0003 | I'm looking for a paper based on the clues below: - Accepted by CVPR 2025 - T... | google-search<br>fetch | google-search | serper_api_key | 是 | - |
| info_search_task_0004 | I'm looking for a paper based on the clues below: - Accepted by ICML 2022 - T... | google-search<br>fetch | google-search | serper_api_key | 是 | - |
| info_search_task_0005 | I'm looking for a city based on the clues below: - the city's football team s... | google-search<br>fetch | google-search | serper_api_key | 是 | - |
| info_search_task_0006 | I'm looking for a city based on the clues below: - The city has a football cl... | google-search<br>fetch | google-search | serper_api_key | 是 | - |
| info_search_task_0007 | A specific person was born in a particular year between 1930 and 1950, both y... | google-search<br>fetch | google-search | serper_api_key | 是 | - |
| info_search_task_0008 | I am looking for a blog that did these things: - Posted in June 2017 that the... | google-search<br>fetch | google-search | serper_api_key | 是 | - |
| info_search_task_0009 | An individual wrote about their experience working as a software developer an... | google-search<br>fetch | google-search | serper_api_key | 是 | - |
| info_search_task_0010 | There is this journal that contains essays written by several religious schol... | google-search<br>fetch | google-search | serper_api_key | 是 | - |
| info_search_task_0011 | There's a poet who lost two family members during a pandemic that occurred be... | google-search<br>fetch | google-search | serper_api_key | 是 | - |
| info_search_task_0012 | To fulfill the requirements of a master’s degree at a university in the USA,... | google-search<br>fetch | google-search | serper_api_key | 是 | - |
| info_search_task_0013 | The artist’s father was a Protestant minister. They had a brief marriage to a... | google-search<br>fetch | google-search | serper_api_key | 是 | - |
| info_search_task_0014 | In Dec 2023, this person said that the least amount of royalties he's earned... | google-search<br>fetch | google-search | serper_api_key | 是 | - |
| info_search_task_0015 | An artist grew up in a town that, up to 2019, had a population of 114000, and... | google-search<br>fetch | google-search | serper_api_key | 是 | - |
| info_search_task_0016 | This individual was born between 1930 and 1940. They were known for works tha... | google-search<br>fetch | google-search | serper_api_key | 是 | - |
| info_search_task_0017 | Before 2023, an individual portrayed a village leader in a film written by mu... | google-search<br>fetch | google-search | serper_api_key | 是 | - |
| info_search_task_0018 | There was a movie that came before 2010 directed by a person who was born in... | google-search<br>fetch | google-search | serper_api_key | 是 | - |
| info_search_task_0019 | An actress born in a city whose flag bears a legendary creature was seen in a... | google-search<br>fetch | google-search | serper_api_key | 是 | - |
| info_search_task_0020 | I am looking for the first and last name of the person (as they were known as... | google-search<br>fetch | google-search | serper_api_key | 是 | - |
| info_search_task_0021 | As of 2023, can you find out the name of the actor who plays the father of th... | google-search<br>fetch | google-search | serper_api_key | 是 | - |
| info_search_task_0022 | An article written in 2023 analyzes the allegorical theme in an animated movi... | google-search<br>fetch | google-search | serper_api_key | 是 | - |
| info_search_task_0023 | A company was co-founded by a former sports athlete in the same year the worl... | google-search<br>fetch | google-search | serper_api_key | 是 | - |
| info_search_task_0024 | Can you identify the wrestler who eliminated a specific competitor from a 30-... | google-search<br>fetch | google-search | serper_api_key | 是 | - |
| info_search_task_0025 | Born in the 1970s, this three-time World Half-Marathon champion, who won thei... | google-search<br>fetch | google-search | serper_api_key | 是 | - |
| info_search_task_0026 | During a specific season of a cricket league held before 2020, a player was a... | google-search<br>fetch | google-search | serper_api_key | 是 | - |
| info_search_task_0027 | Name the landmark in an EU country based on the following details as of Decem... | google-search<br>fetch | google-search | serper_api_key | 是 | - |
| info_search_task_0028 | A 2021 review was written for a historical strategy game that was released ea... | google-search<br>fetch | google-search | serper_api_key | 是 | - |
| info_search_task_0029 | Determine the name of the League of Legends tournament which includes the yea... | google-search<br>fetch | google-search | serper_api_key | 是 | - |
| info_search_task_0030 | I am looking for a specific card in a trading card game. This card was releas... | google-search<br>fetch | google-search | serper_api_key | 是 | - |
| info_search_task_0031 | A video game expansion was released in the 2000s that credited over 200 peopl... | google-search<br>fetch | google-search | serper_api_key | 是 | - |
| info_search_task_0032 | Can you name the video game based on the following details, as of December 20... | google-search<br>fetch | google-search | serper_api_key | 是 | - |
| info_search_task_0033 | I recall viewing a show that had its initial pilot episode broadcast in the 2... | google-search<br>fetch | google-search | serper_api_key | 是 | - |
| info_search_task_0034 | Help me identify the band on the following details available as of Dec 2023:... | google-search<br>fetch | google-search | serper_api_key | 是 | - |
| info_search_task_0035 | Find the title of a song released as a single by Singer A between 2020 and 20... | google-search<br>fetch | google-search | serper_api_key | 是 | - |
| info_search_task_0036 | What is the first and last name of the musician, who was still alive as of De... | google-search<br>fetch | google-search | serper_api_key | 是 | - |
| info_search_task_0037 | A band was formed a year before a United States presidential election, and it... | google-search<br>fetch | google-search | serper_api_key | 是 | - |
| info_search_task_0038 | I want you to identify the name of the artist who strictly fulfills all the b... | google-search<br>fetch | google-search | serper_api_key | 是 | - |
| info_search_task_0039 | Before the year 2000 but after 1995, an article was written to commemorate an... | google-search<br>fetch | google-search | serper_api_key | 是 | - |
| info_search_task_0040 | As of December 2023, identify this historical structure, originally erected b... | google-search<br>fetch | google-search | serper_api_key | 是 | - |
| info_search_task_0041 | An essay was written by a candidate for a PhD in history in 2008 on the subje... | google-search<br>fetch | google-search | serper_api_key | 是 | - |
| info_search_task_0042 | There is a person, who as of 1999 claimed to be twice married. They moved to... | google-search<br>fetch | google-search | serper_api_key | 是 | - |
| info_search_task_0043 | Can you provide me the name of a person acknowledged to provide valuable comm... | google-search<br>fetch | google-search | serper_api_key | 是 | - |
| info_search_task_0044 | What is the name of the program advertised in the article about reducing air... | google-search<br>fetch | google-search | serper_api_key | 是 | - |
| info_search_task_0045 | A person was born in a territory founded in the 13th century by a small princ... | google-search<br>fetch | google-search | serper_api_key | 是 | - |
| info_search_task_0046 | Please provide the first and last name of the individual who meets the follow... | google-search<br>fetch | google-search | serper_api_key | 是 | - |
| info_search_task_0047 | There is an insect that is loved and enjoyed by many people in African countr... | google-search<br>fetch | google-search | serper_api_key | 是 | - |
| info_search_task_0048 | As of 2023, tell me the scientific name of this carnivorous species based on... | google-search<br>fetch | google-search | serper_api_key | 是 | - |
| info_search_task_0049 | Give me the first and last name of the first Prime Minister of this country b... | google-search<br>fetch | google-search | serper_api_key | 是 | - |
| info_search_task_0050 | I am looking for the name of an administrative region (place) in a country th... | google-search<br>fetch | google-search | serper_api_key | 是 | - |
| multi-server_task_google_search_notion_0001 | Yo! I am absolutely obsessed with the player who went 16th in the 2025 NBA dr... | google-search<br>fetch<br>notion | google-search<br>notion | serper_api_key<br>notion_integration_key<br>notion_integration_key_eval<br>source_notion_page_url<br>eval_notion_page_url | 是 | - |
| multi-server_task_google_search_notion_0002 | I am a dedicated follower of Lionel Messi's career. I would like to analyze h... | google-search<br>fetch<br>notion | google-search<br>notion | serper_api_key<br>notion_integration_key<br>notion_integration_key_eval<br>source_notion_page_url<br>eval_notion_page_url | 是 | - |
| multi-server_task_google_search_notion_0003 | As a researcher working on large multimodal models, I need to systematically... | google-search<br>fetch<br>notion | google-search<br>notion | serper_api_key<br>notion_integration_key<br>notion_integration_key_eval<br>source_notion_page_url<br>eval_notion_page_url | 是 | - |
| multi-server_task_google_search_notion_0004 | Hey! I'm a curious boy who really wants to learn about Large Multimodal Model... | google-search<br>fetch<br>notion | google-search<br>notion | serper_api_key<br>notion_integration_key<br>notion_integration_key_eval<br>source_notion_page_url<br>eval_notion_page_url | 是 | - |
| multi-server_task_google_search_notion_0005 | I'm searching for two papers. The first paper was accepted by ACM TIST, the f... | google-search<br>fetch<br>notion | google-search<br>notion | serper_api_key<br>notion_integration_key<br>notion_integration_key_eval<br>source_notion_page_url<br>eval_notion_page_url | 是 | - |
| multi-server_task_playwright_github_0001 | I recently heard from my friend that HuggingFace has created an evaluation su... | playwright<br>github | github | github_token | 否 | api-free 模式当前仍会 fallback 到 live GitHub，因此仍需要 github_token |
| multi-server_task_playwright_github_0002 | I am a new NLP student. My supervisor asked me to improve my coding skills. I... | playwright<br>github | github | github_token | 否 | api-free 模式当前仍会 fallback 到 live GitHub，因此仍需要 github_token |
| multi-server_task_playwright_github_0003 | I am a new NLP student. My supervisor asked me to work on the dataset and res... | playwright<br>github | github | github_token | 否 | api-free 模式当前仍会 fallback 到 live GitHub，因此仍需要 github_token |
| multi-server_task_playwright_github_0004 | I am a new PhD student. My supervisor asked me to work on the GUI grounding p... | playwright<br>github | github | github_token | 否 | api-free 模式当前仍会 fallback 到 live GitHub，因此仍需要 github_token |
| multi-server_task_playwright_github_0005 | I am a big fan of LongVideoBench. Can you help me find the arXiv ID of this p... | playwright<br>github | github | github_token | 否 | api-free 模式当前仍会 fallback 到 live GitHub，因此仍需要 github_token |
| multi-server_task_playwright_google_map_0001 | I am from France. We are planning a trip to Kuala Lumpur, Malaysia. We want y... | google-maps<br>playwright | google-maps | google_cloud_console_api_key | 是 | - |
| multi-server_task_playwright_google_map_0002 | I am a Buddhist. I want to visit the Buddhist temples in Japan, especially th... | google-maps<br>playwright | google-maps | google_cloud_console_api_key | 是 | - |
| multi-server_task_playwright_google_map_0003 | Please help me find a blog post on the website https://www.asiatours.com/. We... | google-maps<br>playwright | google-maps | google_cloud_console_api_key | 是 | - |
| multi-server_task_playwright_google_map_0004 | Please help me find a blog post on the website https://freetourcommunity.com/... | google-maps<br>playwright | google-maps | google_cloud_console_api_key | 是 | - |
| multi-server_task_playwright_google_map_0005 | I met a new friend from England. He told me that he comes from the city with... | google-maps<br>playwright | google-maps | google_cloud_console_api_key | 是 | - |
| multi-server_task_playwright_notion_0001 | I am an undergraduate student in Sport Analysis. One of my courses requires u... | playwright<br>notion | notion | notion_integration_key<br>notion_integration_key_eval<br>source_notion_page_url<br>eval_notion_page_url | 是 | - |
| multi-server_task_playwright_notion_0002 | I am finding a place to live for next month. I will move to Singapore for a n... | playwright<br>notion | notion | notion_integration_key<br>notion_integration_key_eval<br>source_notion_page_url<br>eval_notion_page_url | 是 | - |
| multi-server_task_playwright_notion_0003 | I am an undergraduate student learning Python. I have some questions about 'O... | playwright<br>notion | notion | notion_integration_key<br>notion_integration_key_eval<br>source_notion_page_url<br>eval_notion_page_url | 是 | - |
| multi-server_task_playwright_notion_0004 | I need to arrange a business trip from Singapore to Hong Kong, departing 6 da... | playwright<br>notion | notion | notion_integration_key<br>notion_integration_key_eval<br>source_notion_page_url<br>eval_notion_page_url | 是 | - |
| multi-server_task_playwright_notion_0005 | I need to arrange a business trip from Singapore to Kuala Lumpur, departing 1... | playwright<br>notion | notion | notion_integration_key<br>notion_integration_key_eval<br>source_notion_page_url<br>eval_notion_page_url | 是 | - |
| multi-server_task_weather_google_map_0001 | I live at the Holiday Inn Brooklyn Downtown by IHG in New York City. My frien... | google-maps<br>weather | google-maps | google_cloud_console_api_key | 是 | - |
| multi-server_task_weather_google_map_0002 | I live at the Westin Copley Place, Boston. My friend lives at the Boston Marr... | google-maps<br>weather | google-maps | google_cloud_console_api_key | 是 | - |
| multi-server_task_weather_google_map_0003 | I live in the Empire State Building in New York City. My colleague lives in t... | google-maps<br>weather | google-maps | google_cloud_console_api_key | 是 | - |
| multi-server_task_weather_google_map_0004 | I live in the Terranea Resort, and my colleague lives in the Arizona Biltmore... | google-maps<br>weather | google-maps | google_cloud_console_api_key | 是 | - |
| multi-server_task_weather_google_map_0005 | I live in the Waldorf Astoria Washington DC. My colleague lives in the Hilton... | google-maps<br>weather | google-maps | google_cloud_console_api_key | 是 | - |
| playwright_booking_task_0001 | Help me find a one-way flight from Singapore to Beijing, 5 days from now (If... | playwright<br>date | - | - | 是 | - |
| playwright_booking_task_0002 | I will travel to Singapore 3 days from now (If now is 2025-06-07, then 3 days... | playwright<br>date | - | - | 是 | - |
| playwright_booking_task_0003 | Help me find a round-trip flight from Heathrow to Charles de Gaulle Airport i... | playwright<br>date | - | - | 是 | - |
| playwright_booking_task_0004 | I will travel to Singapore 3 days from now (If today is 2025-06-07, then 3 da... | playwright<br>date | - | - | 是 | - |
| playwright_google_map_task_0001 | I'm from Spain and plant to visit Indonesia. Help me plan an amazing road tri... | playwright<br>date | - | - | 是 | - |
| playwright_google_map_task_0002 | I'm planning an exciting family road trip from Disneyland Paris to the 24 Hou... | playwright<br>date | - | - | 是 | - |
| playwright_google_map_task_0003 | I live in Paiza Sky Residence, Singapore. My friend lives in Symphony Suites,... | playwright<br>date | - | - | 是 | - |
| playwright_google_map_task_0004 | Identify 1 'University' locations in New Delhi, India that are north of the l... | playwright<br>date | - | - | 是 | - |
| playwright_google_map_task_0005 | My partner and I are from Spain and we're planning our romantic honeymoon tri... | playwright<br>date | - | - | 是 | - |
| playwright_google_map_task_0006 | Hey there! As a dad who wants to create the most amazing adventure for my lit... | playwright<br>date | - | - | 是 | - |
| playwright_google_map_task_0007 | I live in Paiza Sky Residence, Singapore. My friend lives in Symphony Suites,... | playwright<br>date | - | - | 是 | - |
| playwright_huggingface_task_0001 | I am new to the field of GUI Agents. I recently read the blog post about the... | playwright<br>date | - | - | 是 | - |
| playwright_huggingface_task_0002 | I am an agent researcher in the field of computer-use. We are working on a pr... | playwright<br>date | - | - | 是 | - |
| playwright_huggingface_task_0003 | I am a big fan of LLM open-source. Can you help me check the total number of... | playwright<br>date | - | - | 是 | - |
| playwright_huggingface_task_0004 | I am a SWE agent researcher, and I am seeking an open-source model for our SW... | playwright<br>date | - | - | 是 | - |
| playwright_huggingface_task_0005 | As a student exploring GUI Agents, I came across an interesting blog post abo... | playwright<br>date | - | - | 是 | - |
| playwright_huggingface_task_0006 | As a professor conducting research in the field of computer-use agents, I hav... | playwright<br>date | - | - | 是 | - |
| playwright_huggingface_task_0007 | Hi! I'm completely new to AI agents and just started learning about them. I h... | playwright<br>date | - | - | 是 | - |
| playwright_huggingface_task_0008 | As a researcher specializing in LLM safety, I am investigating the performanc... | playwright<br>date | - | - | 是 | - |
| playwright_paper_task_0001 | I am a first-year PhD student focusing on Large Multimodal Models. My supervi... | playwright<br>date | - | - | 是 | - |
| playwright_paper_task_0002 | I am a first-year PhD student focusing on NLP. I want to learn about the grea... | playwright<br>date | - | - | 是 | - |
| playwright_paper_task_0003 | I am a first-year PhD student focusing on GUI Grounding. I want to learn abou... | playwright<br>date | - | - | 是 | - |
| playwright_paper_task_0004 | I am a first-year PhD student focusing on large multimodal models. I want to... | playwright<br>date | - | - | 是 | - |
| playwright_paper_task_0005 | I am a first-year PhD student. One of my paper is just accepted by CVPR 2025.... | playwright<br>date | - | - | 是 | - |
| playwright_paper_task_0006 | I am a first-year PhD student focusing on LLMs and I'm super excited to dive... | playwright<br>date | - | - | 是 | - |
| playwright_paper_task_0007 | I'm new to the field of Large Multimodal Models (LMMs) and trying to get up t... | playwright<br>date | - | - | 是 | - |
| playwright_paper_task_0008 | I'm absolutely obsessed with DeepSeek and their incredible work! DeepSeek-R1... | playwright<br>date | - | - | 是 | - |
| playwright_sports_task_0001 | I am a big fan of Lakers in NBA. Can you help me find 3P% of Bronny James and... | playwright<br>date | - | - | 是 | - |
| playwright_sports_task_0002 | I am a big fan of Yueru Li in WNBA. Can you help me find PTS of her in 2024 s... | playwright<br>date | - | - | 是 | - |
| playwright_sports_task_0003 | I am a big fan of Newcastle in Premier League. Can you help me find the total... | playwright<br>date | - | - | 是 | - |
| playwright_sports_task_0004 | I am a big fan of Manchester United in the Premier League. Can you help me fi... | playwright<br>date | - | - | 是 | - |
| playwright_sports_task_0005 | As a passionate football supporter and die-hard fan of the FIFA Club World Cu... | playwright<br>date | - | - | 是 | - |
| playwright_sports_task_0006 | As the BIGGEST Lionel Messi fan in the world, I'm absolutely dying to know ab... | playwright<br>date | - | - | 是 | - |
| playwright_sports_task_0007 | As a HUGE NBA fanatic, I'm absolutely going crazy waiting to find out the res... | playwright<br>date | - | - | 是 | - |
| playwright_sports_task_0008 | As a passionate Gooner and die-hard Arsenal supporter, I'm absolutely despera... | playwright<br>date | - | - | 是 | - |
| yfinance_task_0001 | Hey! I'm super curious about investments and would love your help! Could you... | yfinance<br>calculator | - | - | 是 | - |
| yfinance_task_0002 | Listen here, young one, I need you to figure out what my final value and tota... | yfinance<br>calculator | - | - | 是 | - |
| yfinance_task_0003 | Class, for today's exercise, I want you to analyze an investment scenario. Co... | yfinance<br>calculator | - | - | 是 | - |
| yfinance_task_0004 | Hey, I need you to help me figure out something cool! Say I put $60,000 into... | yfinance<br>calculator | - | - | 是 | - |
| yfinance_task_0005 | My dear fellow, I require you to calculate the final portfolio value and tota... | yfinance<br>calculator | - | - | 是 | - |
| yfinance_task_0006 | I require a comprehensive financial analysis for investment evaluation purpos... | yfinance<br>calculator | - | - | 是 | - |
| yfinance_task_0007 | Hi! I'm working on a finance assignment and need help analyzing some major te... | yfinance<br>calculator | - | - | 是 | - |
| yfinance_task_0008 | I need you to get the latest annual financial statements and calculate the Re... | yfinance<br>calculator | - | - | 是 | - |
| yfinance_task_0009 | As an Apple investor, I need to analyze the company's latest financial perfor... | yfinance<br>calculator | - | - | 是 | - |
| yfinance_task_0010 | As a Microsoft (MSFT) investor, I need to analyze the company's earnings perf... | yfinance<br>calculator | - | - | 是 | - |
| yfinance_task_0011 | I find myself with considerable leisure time today and am rather curious abou... | yfinance<br>calculator | - | - | 是 | - |
| yfinance_task_0012 | Hello! I'm learning about investing and would love to understand how institut... | yfinance<br>calculator | - | - | 是 | - |
| yfinance_task_0013 | Hey! I'm working on a finance project for my BA degree and I need some help w... | yfinance<br>calculator | - | - | 是 | - |
| yfinance_task_0014 | Hey there! I'm building a revolutionary investing startup and I need some kil... | yfinance<br>calculator | - | - | 是 | - |
| yfinance_task_0015 | I'm conducting research for my doctoral dissertation on institutional investm... | yfinance<br>calculator | - | - | 是 | - |
| yfinance_task_0016 | Hey there! I'm working on some analysis for our team and could really use you... | yfinance<br>calculator | - | - | 是 | - |
| yfinance_task_0017 | I'm conducting due diligence on Caterpillar Inc. (CAT) for a potential invest... | yfinance<br>calculator | - | - | 是 | - |
| yfinance_task_0018 | I absolutely love Pepsi and everything about it! As a devoted Pepsi enthusias... | yfinance<br>calculator | - | - | 是 | - |
| yfinance_task_0019 | Hi there! I just started at BlackRock and I'm trying to get up to speed on ou... | yfinance<br>calculator | - | - | 是 | - |
| yfinance_task_0020 | Hi! I'm a financial analyst at Meta and I'm conducting some research on our o... | yfinance<br>calculator | - | - | 是 | - |
| yfinance_task_0021 | Hi there! I'm completely new to investing and trading, but I've been reading... | yfinance<br>calculator | - | - | 是 | - |
| yfinance_task_0022 | Hello dear! I'm absolutely brand new to the world of investing and trading, b... | yfinance<br>calculator | - | - | 是 | - |
| yfinance_task_0023 | Hey! I'm a bachelor student studying finance and I've been learning about dif... | yfinance<br>calculator | - | - | 是 | - |
| yfinance_task_0024 | Hi there! I'm completely new to investing and trading, but I've been reading... | yfinance<br>calculator | - | - | 是 | - |
| yfinance_task_0025 | Hey! I'm a finance student working on a trading strategy assignment. I need t... | yfinance<br>calculator | - | - | 是 | - |
| yfinance_task_0026 | Hi there! I'm a busy mom trying to learn about investing for my family's futu... | yfinance<br>calculator | - | - | 是 | - |
| yfinance_task_0027 | Hey! I'm an undergrad student who's been getting into investing with the mone... | yfinance<br>calculator | - | - | 是 | - |
| yfinance_task_0028 | Hi there! I'm a father of two young kids and I've been working hard to save u... | yfinance<br>calculator | - | - | 是 | - |
| yfinance_task_0029 | Hello! I'm a finance student currently working on my investment portfolio ana... | yfinance<br>calculator | - | - | 是 | - |
| yfinance_task_0030 | Good day! I'm an old-timer looking to make some smart investments for my gran... | yfinance<br>calculator | - | - | 是 | - |
| yfinance_task_0031 | Good day! I'm a careful investor aiming to build a solid foundation for a lon... | yfinance<br>calculator | - | - | 是 | - |
| yfinance_task_0032 | Hey! I'm a finance student currently working on a project about fundamental a... | yfinance<br>calculator | - | - | 是 | - |
| yfinance_task_0033 | Hi there! I'm completely new to investing and finance, and honestly, I'm feel... | yfinance<br>calculator | - | - | 是 | - |
| yfinance_task_0034 | Hi! My dad has been working really hard for our family, and I want to surpris... | yfinance<br>calculator | - | - | 是 | - |
| yfinance_task_0035 | Hi there! My husband and I have been working really hard to build our retirem... | yfinance<br>calculator | - | - | 是 | - |
| yfinance_task_0036 | Hey! I'm a CS student diving into quantitative finance and I'm currently stud... | yfinance<br>calculator | - | - | 是 | - |
| yfinance_task_0037 | Hey there! So I'm an art student and my boyfriend is super into finance - lik... | yfinance<br>calculator | - | - | 是 | - |
| yfinance_task_0038 | Look, I'm broke and desperate here. I've been working dead-end jobs my whole... | yfinance<br>calculator | - | - | 是 | - |
| yfinance_task_0039 | Hey! I'm a huge fan of Lululemon and I've been following their stock religiou... | yfinance<br>calculator | - | - | 是 | - |
| yfinance_task_0040 | Hello darling! I'm absolutely new to the fascinating world of investing and t... | yfinance<br>calculator | - | - | 是 | - |