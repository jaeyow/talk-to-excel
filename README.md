# Environment setup
- use `pyenv` to manage python versions
- use `venv` to manage your virtual environments

```bash
pyenv versions
pyenv install 3.12.2
pyenv local 3.12.2
python -m venv .venv
source .venv/bin/activate
deactivate
```

```bash
chainlit run talk-to-excel-app.py -w
```

## Sample questions for **global_superstore_2016.xlsx**

### Requiring text answers
- Which product sub-category generates the highest total profit?
- Which customer placed the highest number of orders?
- Which region has the highest average shipping delay?
- Does applying higher discounts lead to lower profits?
- Which product category has the highest average shipping cost?

### Requiring plots
- Plot the distribution of sales across different regions. Use a bar chart.
- Using a bar chart, compare total profit across product categories.
- Plot the total sales trend over time. Use a line graph.
- Who are the top 10 customers in terms of sales? Plot using a horizontal bar chart.
- How are sales distributed across different regions? Use a pie chart then add an appropriate title. Show top ten only.
- How do different discount levels affect profit margins? Is there a discount threshold where profitability drops significantly? Visualise this.