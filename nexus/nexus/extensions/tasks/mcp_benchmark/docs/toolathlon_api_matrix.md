# Toolathlon 题目 API / 凭据依赖矩阵

由 `uv run python -m nexus.extensions.tasks.mcp_benchmark.toolathlon_selection matrix` 生成。

- `依赖 MCP 服务`: 这道题会加载的 MCP 服务集合。
- `需要额外凭据的服务`: 在 `original` 模式下需要额外准备凭据、OAuth 文件或第三方账号配置的服务。
- `api-free 可跑`: 在缺少 original 额外凭据时，当前 benchmark 是否还能以 best-effort 方式运行这道题。

| case_id | 题目摘要 | 依赖 MCP 服务 | 需要额外凭据的服务 | 凭据字段名 | api-free 可跑 | api-free 说明 |
| --- | --- | --- | --- | --- | --- | --- |
| ab-testing | The A/B test for our new homepage has concluded, and the raw clickstream data... | google-cloud<br>filesystem | google-cloud | gcp_project_id<br>gcp_service_account_path | 否 | api-free 模式当前未打通 google-cloud 链路，仍需要上游 GCP project / service-account 配置 |
| academic-pdf-report | I am conducting research on papers for an AI Conference. For each paper, plea... | playwright_with_chunk<br>excel<br>terminal<br>arxiv_local<br>pdf-tools<br>fetch | - | - | 是 | - |
| academic-warning | The academic affairs system requires an automated academic warning mechanism.... | google-cloud<br>excel<br>filesystem | google-cloud | gcp_project_id<br>gcp_service_account_path | 否 | api-free 模式当前未打通 google-cloud 链路，仍需要上游 GCP project / service-account 配置 |
| add-bibtex | Please help me add the following article to the ref.bib file in the workspace... | scholarly<br>playwright_with_chunk<br>filesystem<br>terminal<br>fetch | - | - | 是 | - |
| apply-phd-email | Please check my inbox for an email from 'kaiming', and then help me submit th... | filesystem<br>memory<br>emails<br>terminal<br>pdf-tools | - | - | 是 | - |
| arrange-workspace | Please help me organize the files in my workspace. The file structure should... | filesystem<br>terminal<br>pdf-tools<br>excel | - | - | 是 | - |
| canvas-arrange-exam | It is exam week now, and today is January 16, 2025. Check the announcements i... | canvas<br>emails<br>memory<br>terminal | - | - | 是 | - |
| canvas-art-manager | This is admin3, a registrar from XXX University. The university has sent me t... | filesystem<br>canvas<br>terminal<br>emails | - | - | 是 | - |
| canvas-art-quiz | I am the teacher in charge of the Art History course at school. Please help m... | filesystem<br>canvas<br>terminal | - | - | 是 | - |
| canvas-do-quiz | My personal information is stored in memory. Check which unfinished course qu... | memory<br>canvas | - | - | 是 | - |
| canvas-homework-grader-python | Your task is to check your email for homework2 submissions and grade on Canva... | canvas<br>filesystem<br>terminal<br>emails | - | - | 是 | - |
| canvas-list-test | My personal information is all stored in memory. Based on the course informat... | canvas<br>memory | - | - | 是 | - |
| canvas-new-students-notification | Several new students who have changed majors will be taking courses this seme... | filesystem<br>terminal<br>canvas | - | - | 是 | - |
| canvas-submit-late-work | My name is Zachary Edwards, and I’m a forgetful person. I often complete my a... | canvas<br>memory<br>filesystem<br>emails | - | - | 是 | - |
| cooking-guidance | Using the ingredients currently available (listed in the ingredients.csv file... | filesystem<br>howtocook | - | - | 是 | - |
| course-assistant | I am the teaching assistant for the NLP course. The final presentation assign... | excel<br>emails<br>filesystem<br>terminal | - | - | 是 | - |
| course-schedule | My personal information is stored in Memory. The workspace contains my course... | filesystem<br>memory<br>excel<br>pdf-tools<br>fetch | - | - | 是 | - |
| courses-ta-hws | Please find all files in the workspace that are related to the "Operating Sys... | terminal<br>excel<br>filesystem | - | - | 是 | - |
| cvpr-research | I am a doctoral student in computer vision and will graduate next year. My ca... | filesystem<br>fetch<br>playwright_with_chunk | - | - | 是 | - |
| dataset-license-issue | Someone just filed an issue in my Annoy-DataSync project asking about the lic... | huggingface<br>github<br>terminal<br>fetch | huggingface<br>github | huggingface_token<br>github_token | 否 | api-free 模式当前仍会 fallback 到 live Hugging Face，因此仍需要 huggingface_token<br>api-free 模式当前仍会 fallback 到 live GitHub，因此仍需要 github_token |
| detect-revised-terms | Please help me read new_law.pdf and each case judgment document, identify any... | filesystem<br>pdf-tools | - | - | 是 | - |
| dietary-health | I am a male university student, and my detailed body information is in `body.... | filesystem<br>howtocook<br>excel<br>terminal | - | - | 是 | - |
| email-paper-homepage | Update my personal homepage according to the following rules: - For papers cu... | emails<br>github | github | github_token | 否 | api-free 模式当前仍会 fallback 到 live GitHub，因此仍需要 github_token |
| excel-data-transformation | Please convert the file `Household_Appliances.xlsx` from its current two-dime... | excel<br>filesystem<br>terminal | - | - | 是 | - |
| excel-market-research | Based on the provided file `Market_Data.xlsx`, calculate the year-over-year g... | excel<br>filesystem<br>terminal | - | - | 是 | - |
| experiments-recordings | In the table of the Notion page `mcp_experiments_recordings`, based on the hi... | notion<br>wandb<br>terminal<br>filesystem | notion<br>wandb | notion_integration_key<br>notion_integration_key_eval<br>source_notion_page_url<br>eval_notion_page_url<br>wandb_api_key | 否 | api-free 模式当前仍会 fallback 到 live Weights & Biases，因此仍需要 wandb_api_key |
| fillout-online-forms | My good friend Alex and I both want to attend the Freshmen Welcome Party and... | playwright_with_chunk<br>memory<br>filesystem | - | - | 是 | - |
| filter-low-selling-products | Please help me check which products are in my shop, filter out those that hav... | woocommerce<br>filesystem<br>emails | - | - | 是 | - |
| find-alita-paper | I'm looking for a paper on arxiv related to agentic reasoning. I only remembe... | arxiv_local<br>filesystem<br>scholarly | - | - | 是 | - |
| flagged-transactions | Perform anomaly detection on high-net-worth clients’ transactions: Extract th... | google-cloud<br>excel<br>terminal<br>filesystem | google-cloud | gcp_project_id<br>gcp_service_account_path | 否 | api-free 模式当前未打通 google-cloud 链路，仍需要上游 GCP project / service-account 配置 |
| game-statistics | The end-of-day settlement time for the game has arrived, and we need to proce... | google-cloud<br>terminal | google-cloud | gcp_project_id<br>gcp_service_account_path | 否 | api-free 模式当前未打通 google-cloud 链路，仍需要上游 GCP project / service-account 配置 |
| gdp-cr5-analysis | Please read the table titled "GDP2022" in Google Sheets and calculate the CR5... | google_sheet<br>playwright_with_chunk<br>fetch | google_sheet | google_oauth_credentials_file | 是 | - |
| git-bug-hunt | In the `LUFFY` Git repository within the workspace, we've identified a seriou... | git<br>terminal<br>filesystem<br>emails | - | - | 是 | - |
| git-milestone | In recent days, the total number of repos on GitHub has reached 1B, which is... | filesystem<br>terminal<br>fetch | - | - | 是 | - |
| git-repo | I'm trying to reproduce the experiments from a deep learning paper (paper.pdf... | github<br>filesystem<br>pdf-tools | github | github_token | 否 | api-free 模式当前仍会 fallback 到 live GitHub，因此仍需要 github_token |
| hk-top-conf | I'm a third-year undergraduate student majoring in Computer Science, and I pl... | filesystem<br>terminal<br>playwright_with_chunk<br>fetch | - | - | 是 | - |
| huggingface-upload | Please scan the workspace folder, pick the model checkpoint with the highest... | filesystem<br>terminal<br>huggingface | huggingface | huggingface_token | 否 | api-free 模式当前仍会 fallback 到 live Hugging Face，因此仍需要 huggingface_token |
| identify-all-songs | Please help me find the YouTube video titled ' Top hits 2024 playlist Trendin... | filesystem<br>youtube-transcript<br>playwright_with_chunk<br>fetch | - | - | 是 | - |
| imagenet | I am a graduate student in computer vision, and I have several papers on imag... | filesystem<br>pdf-tools | - | - | 是 | - |
| inter-final-performance-analysis | I would like to know Inter Milan's performance in the 2023 and 2025 Champions... | filesystem<br>playwright_with_chunk<br>google_sheet<br>terminal<br>fetch | google_sheet | google_oauth_credentials_file | 是 | - |
| interview-report | As an HR, I've just finished a day of interviews and have several candidates'... | filesystem<br>word | - | - | 是 | - |
| inventory-sync | I am a WooCommerce store owner, and my shop has warehouses in the following c... | woocommerce<br>filesystem | - | - | 是 | - |
| investment-decision-analysis | You are an investment analysis assistant. Please use Yahoo Finance to perform... | yahoo-finance<br>google_sheet<br>excel<br>filesystem | google_sheet | google_oauth_credentials_file | 是 | - |
| invoice-org | Please help me read the files in my personal workspace, find all of my invoic... | pdf-tools<br>filesystem<br>yahoo-finance<br>excel | - | - | 是 | - |
| ipad-edu-price | You are an Apple education discount analysis assistant, and I am a university... | yahoo-finance<br>playwright_with_chunk<br>fetch | - | - | 是 | - |
| k8s-deployment-cleanup | Find all Deployments in namespaces with names starting with 'dev-' that are r... | k8s<br>emails<br>pdf-tools<br>filesystem | - | - | 是 | - |
| k8s-mysql | I need to debug the **f1** database in the `data` namespace of my Kubernetes... | k8s<br>filesystem | - | - | 是 | - |
| k8s-pr-preview-testing | Help me deploy the `feature/pr-123` branch of the `Toolathlon-Archive/SimpleS... | terminal<br>filesystem<br>playwright_with_chunk<br>k8s | - | - | 是 | - |
| k8s-redis-helm-upgrade | Upgrade the 'redis' chart from the 'bitnami' Helm repository in the shared-se... | k8s<br>terminal<br>filesystem | - | - | 是 | - |
| k8s-safety-audit | Perform a security audit on the Kubernetes cluster and append the results to... | k8s<br>google_sheet<br>filesystem | google_sheet | google_oauth_credentials_file | 是 | - |
| landing-task-reminder | Today we have 3 new employees joining us, and each employee's information is... | emails<br>snowflake<br>pdf-tools<br>filesystem | snowflake | snowflake_account<br>snowflake_role<br>snowflake_user<br>snowflake_private_key_file | 是 | - |
| language-school | You are preparing a graduate application report for a student who wants to ap... | filesystem<br>playwright_with_chunk<br>excel<br>fetch<br>terminal | - | - | 是 | - |
| latex-prompt-box | Only edit the TODO part at the end of the appendix within my paper Latex proj... | filesystem<br>arxiv-latex<br>terminal | - | - | 是 | - |
| live-transactions | A suspicious transaction (ID: T8492XJ3) has been detected in the live_transac... | google-cloud<br>filesystem | google-cloud | gcp_project_id<br>gcp_service_account_path | 否 | api-free 模式当前未打通 google-cloud 链路，仍需要上游 GCP project / service-account 配置 |
| llm-training-dataset | I've been learning about large-scale language models recently, and I've decid... | playwright_with_chunk<br>google_sheet<br>scholarly<br>fetch | google_sheet | google_oauth_credentials_file | 是 | - |
| logical-datasets-collection | Hello! I am a PhD student engaged in deep learning research. Recently, my col... | filesystem<br>scholarly<br>arxiv_local<br>pdf-tools<br>fetch | - | - | 是 | - |
| machine-operating | A production line in the factory streams real-time data through IoT sensors i... | google-cloud<br>filesystem<br>excel | google-cloud | gcp_project_id<br>gcp_service_account_path | 否 | api-free 模式当前未打通 google-cloud 链路，仍需要上游 GCP project / service-account 配置 |
| meeting-assign | Please read the content from localhost:30137 and summarize the information, p... | fetch<br>emails<br>playwright_with_chunk<br>filesystem | - | - | 是 | - |
| merge-hf-datasets | Could you please find the following datasets on Hugging Face, with their corr... | huggingface<br>terminal<br>filesystem | huggingface | huggingface_token | 否 | api-free 模式当前仍会 fallback 到 live Hugging Face，因此仍需要 huggingface_token |
| mrbeast-analysis | As a novice video content creator, I want to analyze the video style and patt... | youtube<br>filesystem<br>excel<br>youtube-transcript | youtube | google_cloud_console_api_key | 是 | - |
| music-analysis | A music streaming platform wants to accurately identify the 1940s pop singles... | excel<br>google_sheet<br>terminal | google_sheet | google_oauth_credentials_file | 是 | - |
| nhl-b2b-analysis | Conduct a detailed analysis of the NHL 2024–2025 schedule (in spreadsheet `NH... | google_sheet<br>filesystem<br>terminal | google_sheet | google_oauth_credentials_file | 是 | - |
| notion-find-job | You are a professional job search assistant. You need to manage a Notion page... | google_map<br>notion<br>emails<br>playwright_with_chunk | google_map<br>notion | google_cloud_console_api_key<br>notion_integration_key<br>notion_integration_key_eval<br>source_notion_page_url<br>eval_notion_page_url | 是 | - |
| notion-hr | Please update the candidate information on the HR Record subpage of the Notio... | filesystem<br>emails<br>notion<br>pdf-tools | notion | notion_integration_key<br>notion_integration_key_eval<br>source_notion_page_url<br>eval_notion_page_url | 是 | - |
| notion-movies | You are a professional notion page update assistant. You need to update a not... | playwright_with_chunk<br>notion<br>fetch | notion | notion_integration_key<br>notion_integration_key_eval<br>source_notion_page_url<br>eval_notion_page_url | 是 | - |
| notion-personal-website | You need to update a notion page named `Colley Whisson` based on the informat... | filesystem<br>word<br>notion | notion | notion_integration_key<br>notion_integration_key_eval<br>source_notion_page_url<br>eval_notion_page_url | 是 | - |
| nvidia-market | Your task is to analyze NVIDIA’s institutional ownership trends over 8 quarte... | yahoo-finance<br>filesystem<br>terminal<br>playwright_with_chunk<br>excel<br>fetch | - | - | 是 | - |
| nvidia-stock-analysis | Please help me analyze the ownership and market trends of NVIDIA (NVDA). In t... | yahoo-finance<br>filesystem<br>terminal<br>excel | - | - | 是 | - |
| oil-price | I have prepared two data tables in the Notion `Oil Price` page. Please fetch... | yahoo-finance<br>notion<br>filesystem<br>terminal | notion | notion_integration_key<br>notion_integration_key_eval<br>source_notion_page_url<br>eval_notion_page_url | 是 | - |
| paper-checker | Help me check my LaTeX paper draft for citation and reference issues. I have... | filesystem<br>terminal | - | - | 是 | - |
| payable-invoice-checker | Help me automatically update the two existing tables in the PURCHASE_INVOICE... | emails<br>pdf-tools<br>filesystem<br>snowflake | snowflake | snowflake_account<br>snowflake_role<br>snowflake_user<br>snowflake_private_key_file | 是 | - |
| personal-website-construct | Please fork https://github.com/academicpages/academicpages.github.io and rena... | memory<br>github | github | github_token | 否 | api-free 模式当前仍会 fallback 到 live GitHub，因此仍需要 github_token |
| ppt-analysis | I'm a third-year computer science student. I was absent from the last Compile... | pptx<br>filesystem<br>pdf-tools | - | - | 是 | - |
| price-comparison | As a member of the market analysis team, you are urgently tasked with analyzi... | filesystem<br>terminal<br>pdf-tools<br>google-cloud | google-cloud | gcp_project_id<br>gcp_service_account_path | 否 | api-free 模式当前未打通 google-cloud 链路，仍需要上游 GCP project / service-account 配置 |
| privacy-desensitization | I need to perform privacy data desensitization on my documents. The files may... | filesystem<br>terminal | - | - | 是 | - |
| profile-update-online | I’ve recently uploaded two articles (B-Star and SimpleRL-Zoo). Please update... | filesystem<br>playwright_with_chunk<br>scholarly<br>arxiv_local<br>pdf-tools | - | - | 是 | - |
| quantitative-financial-analysis | Please create a Google Sheet named '2025_Market_Data' with a worksheet titled... | filesystem<br>yahoo-finance<br>google_sheet<br>notion<br>terminal | google_sheet<br>notion | google_oauth_credentials_file<br>notion_integration_key<br>notion_integration_key_eval<br>source_notion_page_url<br>eval_notion_page_url | 是 | - |
| reimbursement-form-filler | I'm an accountant in the R&D department of a tech company, and I need your he... | filesystem<br>excel<br>pdf-tools<br>terminal | - | - | 是 | - |
| sales-accounting | Today is January 18, 2024. Please only add the transaction records I have tol... | memory<br>excel<br>filesystem | - | - | 是 | - |
| search-ca-school | Please help me prepare a graduate application report for a student currently... | filesystem<br>google_map<br>playwright_with_chunk<br>fetch | google_map | google_cloud_console_api_key | 是 | - |
| set-conf-cr-ddl | Check my emails from the past day. If any mention the COML conference main-tr... | emails<br>google_calendar | google_calendar | google_oauth_credentials_file | 是 | - |
| shopping-helper | I'm looking to buy furniture for my new house and want to buy a sofa on amazo... | filesystem<br>playwright_with_chunk | - | - | 是 | - |
| sla-timeout-monitor | Identify the tickets in the database that have exceeded the initial response... | emails<br>snowflake<br>filesystem<br>pdf-tools<br>terminal | snowflake | snowflake_account<br>snowflake_role<br>snowflake_user<br>snowflake_private_key_file | 是 | - |
| stock-build-position | I am preparing to open a stock position today. `stock.xlsx` contains the comp... | excel<br>yahoo-finance<br>terminal | - | - | 是 | - |
| student-interview | I am a professor, recently some students have sent me resumes by eamils, and... | emails<br>google_calendar | google_calendar | google_oauth_credentials_file | 是 | - |
| subway-planning | I am staying at the Singapore Mobility Gallery and would like to travel to Ch... | google_map<br>filesystem<br>playwright_with_chunk<br>fetch | google_map | google_cloud_console_api_key | 是 | - |
| sync-todo-to-readme | In branch dev of **my** LUFFY repo, I have written some new code that I have... | git<br>github | github | github_token | 否 | api-free 模式当前仍会 fallback 到 live GitHub，因此仍需要 github_token |
| task-tracker | Please find all developers' branches in the `BenchTasksCollv3` project for th... | github<br>notion | github<br>notion | github_token<br>notion_integration_key<br>notion_integration_key_eval<br>source_notion_page_url<br>eval_notion_page_url | 否 | api-free 模式当前仍会 fallback 到 live GitHub，因此仍需要 github_token |
| train-ticket-plan | My friend and I have planned a trip to qufu(曲阜). We will depart on the evenin... | filesystem<br>rail_12306<br>terminal<br>fetch | - | - | 是 | - |
| travel-exchange | I (Andrew) and six friends (Lau, Chen, Diana, Elena, Frank, and Grace) travel... | terminal<br>filesystem<br>yahoo-finance | - | - | 是 | - |
| travel-expense-reimbursement | In my workspace, employees have uploaded their expense claim materials. You n... | emails<br>snowflake<br>pdf-tools<br>filesystem | snowflake | snowflake_account<br>snowflake_role<br>snowflake_user<br>snowflake_private_key_file | 是 | - |
| trip-adviser | I want to depart from the Nihonbashi Exit of Tokyo Station, with the goal of... | google_map<br>fetch<br>filesystem<br>playwright_with_chunk | google_map | google_cloud_console_api_key | 是 | - |
| trip-itinerary-generator | I plan to go on vacation to Paris soon. Please help me create a two-day Paris... | filesystem<br>google_map<br>playwright_with_chunk | google_map | google_cloud_console_api_key | 是 | - |
| university-course-selection | There is a class schedule for the first semester of the 2025-26 academic year... | filesystem<br>pdf-tools<br>terminal<br>excel | - | - | 是 | - |
| update-material-inventory | Please help me monitor new paid orders in WooCommerce, retrieve the SKU and q... | google_sheet<br>woocommerce | google_sheet | google_oauth_credentials_file | 是 | - |
| upenn-campus-route | I am currently at the University of Pennsylvania (UPenn) Bookstore. I would l... | filesystem<br>terminal<br>google_map<br>fetch | google_map | google_cloud_console_api_key | 是 | - |
| verl-dataset | I'm trying to use the Verl framework to replicate DeepSeek-R1's "aha moment"... | huggingface<br>filesystem<br>fetch | huggingface | huggingface_token | 否 | api-free 模式当前仍会 fallback 到 live Hugging Face，因此仍需要 huggingface_token |
| vlm-history-completer | Please help me organize and complete the development history of Vision Langua... | playwright_with_chunk<br>google_sheet<br>arxiv_local<br>huggingface<br>fetch | google_sheet<br>huggingface | google_oauth_credentials_file<br>huggingface_token | 否 | api-free 模式当前仍会 fallback 到 live Hugging Face，因此仍需要 huggingface_token |
| wandb-best-score | Analyze the wandb project https://wandb.ai/mluo/deepscaler-1.5b?nw=nwusermluo... | wandb<br>filesystem<br>terminal | wandb | wandb_api_key | 否 | api-free 模式当前仍会 fallback 到 live Weights & Biases，因此仍需要 wandb_api_key |
| wandb-shortest-length | Analyze the wandb project https://wandb.ai/mluo/deepscaler-1.5b?nw=nwusermluo... | wandb<br>filesystem<br>terminal<br>excel | wandb | wandb_api_key | 否 | api-free 模式当前仍会 fallback 到 live Weights & Biases，因此仍需要 wandb_api_key |
| woocommerce-customer-survey | Use the WooCommerce to check all order status updates for customers marked as... | woocommerce<br>emails<br>google_forms<br>filesystem | google_forms | google_oauth_credentials_file | 是 | - |
| woocommerce-new-product | As an operator of the WooCommerce e-commerce platform, you need to send email... | woocommerce<br>filesystem<br>emails | - | - | 是 | - |
| woocommerce-new-welcome | Please check all customers who have completed their first order in our store... | woocommerce<br>filesystem<br>terminal<br>google-cloud<br>emails | google-cloud | gcp_project_id<br>gcp_service_account_path | 否 | api-free 模式当前未打通 google-cloud 链路，仍需要上游 GCP project / service-account 配置 |
| woocommerce-product-recall | Remove products of a specified model from your WooCommerce store and recall t... | woocommerce<br>emails<br>google_forms<br>filesystem | google_forms | google_oauth_credentials_file | 是 | - |
| woocommerce-stock-alert | You need to read the inventory levels of WooCommerce products, check the curr... | woocommerce<br>google_sheet<br>emails<br>filesystem | google_sheet | google_oauth_credentials_file | 是 | - |
| woocommerce-update-cover | Update the main product image based on WooCommerce order data, setting the im... | woocommerce | - | - | 是 | - |
| yahoo-analysis | Evaluate the accuracy of historical analyst ratings for NVIDIA (NVDA) and App... | yahoo-finance<br>filesystem<br>terminal | - | - | 是 | - |
| youtube-repo | Please read the video list at https://www.youtube.com/playlist?list=PLyzTA8ce... | youtube<br>youtube-transcript<br>filesystem<br>fetch<br>github | youtube<br>github | google_cloud_console_api_key<br>github_token | 否 | api-free 模式当前仍会 fallback 到 live GitHub，因此仍需要 github_token |