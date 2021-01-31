1. Steps to submit to online evaluation
	0. Prep
		A. Do this for files with @ permission if they exist. Check by: ls -l
			>xattr -c
			>xattr -cr

		B. Remove .DS_Store if it exists. Check by: 
			>ls -a
			>rm -rf .DS_Store

	1. Inside the pred directory, create .tar.gz:
		
		>tar -cvzf files.tar.gz .

		For GE13, use csf3 to create the tar file.
		- zip the pred file, replace some files to have the characters below:
			PMC-3218875-06-Negative_regulation_of_NF-κB_signalling_in_RA.a2
		- transfer to csf3
		- unzip the file
		- tar -cvzf files.tar.gz .

	2. Then submit the .tar.gz online.
2. How to debug problems in dev/train predictions like this:
	PMID not in annotation: PMID-18667841
	<_io.TextIOWrapper name='<stderr>' mode='w' encoding='UTF-8'> # NOTE: SKIPPING ../../epoch-output-folder/sbm_tees_gold_cg13_08_27/pred/PMID-21536653.a2, READ FAILED: RESULTS WILL NOT BE VALID FOR THE FULL DATASET!
	<_io.TextIOWrapper name='<stderr>' mode='w' encoding='UTF-8'> 
	....
	....
	NOTE: evaluation succeeded only for 91/100 documents. Results will not be valid for the whole dataset.

	Answer: 

Profiling code
- uncomment the call to the predict function in the SBNN file
