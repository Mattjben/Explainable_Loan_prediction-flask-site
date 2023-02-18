# Explainable Loan Prediction
The aim of this project was to design and implement a loan prediction app that both predicted the risk level of customers based on their banking data and outputted a detailed explanation to prevent unethical buissness practices. 

This project can be seperated into five phases: 
1. Data wrangling 
2. Model testing and analysis (jupyter-notbook directory)
3. Front end development 
4. Web hosting (orginally hosted on heroku , however link has expired as of 2023)
5. Report summmary 



## **Members**
- Alex Peter Thomas
- Isxaq Warsame
- Simran Sidhanti
- John Fergus Murrowood
- Matthew John Bentham
- Udit Dinesh

## **Important Links**
- [Trello WorkSpace](https://trello.com/b/3AZXMNnj)
- [Trello PM Board](https://trello.com/invite/b/3AZXMNnj/fcf43e198a5f7ae7fee3503644bba815/csids-pm-board)
- [GitHub Link](https://github.com/IwVr/CSIDS-Finance.git)

## **Notes** 
- Please ensure you use proper git methods
- For adding a feature create your own branch and the make a pull request so we can review the changes
- Avoid commiting directly to the main branch
- Ensure the commit messages follow proper rules 


## **Git Contents & Structure**
- "Datasets" Folder holds all the data we use
- "Jupyter-Notebooks" Folder Holds all the notebooks we use & also the back-end code base we use
- "App-Design" Folder Holds all the front-end Stuff such as the app and the code base for it
- "Docs" Folder holds all the documents in the folder such as the draft and so forth. Make some subfolders for it.

## **Dataset description:**
**Definitions and special values:**              
**Inquiry:** Hard credit inquiries which occur when a creditor has requested to look at your credit file (e.g., when applying for new credit card)  large amounts of inquires on a file could indicate uncertainty /instability (e.g. filing for multiple cards at once)

**Inq excl 7days:** generally, multiple inquires made within the same 7 days are attributed to price comparison shopping and therefore aren’t an indication of instability 

**Derogatory comment:** negative item in your credit reports (e.g. caused by 180+ days late payments , creditor takes possession of property due to non-payment , files for bankruptcy etc.) 

**Delinquency:** refers to a payment received some period past its due date
Instalment trade: credit agreements you make to pay an account over time (e.g., home loan)

**Revolving trade:** a credit agreement that provides you with a credit limit you’re allowed to use and pay back over time. 

**-9:** No Bureau Record or No Investigation (Missing value)           
**-8:** No Usable/Valid Accounts Trades or Inquiries - inactive/very old account                       
**-7:** Condition not Met (e.g., someone has no delinquencies in the last 12 months, different form 0 as 0 means no delinquencies ever)  - can convert to 0 if this distinction is unnecessary   

Variable names | Description 
--- | ---
ExternalRiskEstimate|	? 
MSinceOldestTradeOpen|	Months since oldest approved credit agreement 
MSinceMostRecentTradeOpen|	Months since last approved credit agreement
AverageMInFile|	Average Months in File
NumSatisfactoryTrades|	number of credit agreements on a consumer credit bureau report with on-time payments
NumTrades60Ever2DerogPubRec|	the number of credit agreements on a credit bureau report that record a payment received 60 days past its due date
NumTrades90Ever2DerogPubRec|	the number of credit agreements on a credit bureau report that record a payment received 90 days past its due date
PercentTradesNeverDelq|	Percentage of credit agreements on a consumer credit bureau report with on-time payments
MSinceMostRecentDelq|	Months since most recent overdue payment
MaxDelq2PublicRecLast12M|	Maximum number of credit agreements with overdue payments or derogatory comments in the last 12 months. 
MaxDelqEver|	Maximum number of credit agreements with overdue payments.
NumTotalTrades|	Total number of credit agreements 
NumTradesOpeninLast12M|	Total number of credit agreements open in the last 12 months 
PercentInstallTrades|	Percent Installment Trades
MSinceMostRecentInqexcl7days|	Months Since Most Recent inquiry, excluding the last 7 days 
NumInqLast6M|	Number of inquires in the last 6 months 
NumInqLast6Mexcl7days|	Number of inquires in the last 6 months, excluding the last 7 days 
NetFractionRevolvingBurden|	 RevolvingBurden = (portion of credit card spending that goes unpaid at the end of a billing cycle)/ credit limit 
NetFractionInstallBurden|	InstallBurden = (portion of loan that goes unpaid at the end of a billing cycle)/ (monthly instalment to be paid) 
NumRevolvingTradesWBalance|	Number of revolving trades currently not fully paid off 
NumInstallTradesWBalance|	Number of instalment trades currently not fully paid off
NumBank2NatlTradesWHighUtilization|	Number of bank trades with high utilization ratio utilization ratio: percentage of available credit you're using on your revolving credit accounts (higher means your close to maxing out)
PercentTradesWBalance|	Number of trades currently not fully paid off
