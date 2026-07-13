# Chattermill LLM Challenge

At Chattermill we analyse customer feedback at scale, helping businesses understand not just how customers feel, but **what** they are talking about and why it matters. One of the core challenges we work on is extracting structured, meaningful insight from raw, unstructured feedback text. Customer feedback naturally organises around a set of high-level topics (things like account access, app performance, or customer service), which we call themes. That is where you come in.

### The Goal

We would like you to build a pipeline that takes a dataset of customer feedback comments and produces structured insights from it. Your pipeline should:

1. **Extract aspects** — use LLMs to identify and extract meaningful units of information from each feedback comment, along with the sentiment expressed towards each aspect (for example, topics customers are mentioning, the nature of their experience, relevant attributes of the product or service, and whether the sentiment is positive, negative, or neutral). Aspects should be granular and specific, much more precise than the high-level themes provided in `themes.json`.
2. **Create insights** — group, model, or otherwise structure the extracted units into coherent patterns or clusters. How you do this is entirely up to you.
3. **Map to themes** — map your discovered insights to the high-level Theme Structure provided in `themes.json`. Each theme belongs to an even higher-level Category; your mapping should reflect that hierarchy.
4. **Evaluate your output** — assess the quality of your pipeline and its results. This step is important: think carefully about what "good" looks like here and how you would measure it.

### Data

`data/feedback.csv` contains 5,000 real customer feedback comments. There is a single column: `comment`. There are no labels.

### Theme Structure

`themes.json` provides a two-level hierarchy of Categories and Themes. Use this as your reference frame for step 3. You are not expected to invent new top-level categories; map to what is there.

### LLM API Budget

We will provide you with **$20 of API credit** to spend on this task. Use it thoughtfully, as efficient use of the budget is itself a signal.

### Expected Output

Please submit your solution as a zip file using the Greenhouse link provided by our talent team. Your submission should include:

* Your code (notebooks, scripts, or a combination). We value clear, modular, and well-structured code as much as the ML approach itself.
* The output of each pipeline stage (extracted aspects, insight descriptions, theme mapping results)
* Your evaluation: methodology, metrics or criteria, and results
* A `README.md` explaining your process, the decisions you made, and what you found. It can be in the format of a short report, blog post, or tutorial. **The file should contain full instructions for rerunning your code.**

### Notes

* You should be able to complete the project within a few hours, not days.
* Feel free to work in whatever language or framework you are most comfortable in, but we prefer Python for this type of task.
* You may use any combination of LLMs, classical algorithms, and statistical models — or just one. There is no prescribed approach.
* **The emphasis is on clear thinking and well-documented decisions**. Explain *why* you made the choices you did. We value concise, well-reasoned write-ups over exhaustive ones, as they tell us more about how you think and how well you can communicate and advance your ideas.
* Treat your code as if it were going into a production pipeline; clarity, modularity, and reproducibility matter.
* If you pass this stage, you will be asked to present your project in a short presentation (~10 minutes). We are interested in your thinking, the decisions you made along the way, and the overall pipeline design.

### Bonus Points

* Clear, readable code and good naming of methods
* A thoughtful and well-reasoned evaluation approach
* Any interesting or surprising insights you surface from the data

### Get in Touch

If something is not clear, feel free to submit an issue to this repo or drop us an email at nlp@chattermill.io.
