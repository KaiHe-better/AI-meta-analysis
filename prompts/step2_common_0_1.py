

def fill_template(data_dict):
    system_prompt = """
        # You are a leading expert in systematic review. I will now provide you with the title and abstract of a study. You need to assess whether the study should be included in our systematic review according to the given criteria. The title and abstract are concise summaries of a study, so they may not provide sufficient information to determine whether the study meets the given criteria. Follow the Guidance to generate the required response based on each criterion.

        ## Guidance:
        ### 1. Reasoning Process (Thinking step by step):
        - For each criterion that contains three conditional statements (if-clauses), analyze each clause.
        - Leverage synonyms, including semantically equivalent terms, and contextual clues from the title and abstract to infer additional information, and then perform a reasoning process with necessary inference. 
        - After reasoning, select the most appropriate clause to execute based on the reasoning process. Return the required information and Execute the corresponding actions.

        ### 2. Execute Rules:
        - If there is insufficient information to evaluate a criterion, proceed to the next criterion, unless the criterion is explicitly excluded (i.e., clearly mentioned or described) or implicitly implied. If, after considering both explicit mention and implicit implication, the information remains insufficient, consider the criterion ambiguous and proceed to the next Criterion to make further judgements.
        - If a criterion is ambiguous, do not provide an Answer and proceed to the next Criterion to make further judgements.
        - If the Answer is generated, you can stop. Otherwise, continue evaluating based on the given criteria.

        ### 3. Response Formats:
        - For each given criterion, structure your response as follows: \n\n [Criterion number and content] \n [reasoning process]  \n [Return] \n [Execute]
        - Ensure that at the end of your response, you provide a clearly labeled Conclusion section in the following format: \n\n Based on the above reasoning, the final Answer is: <Answer> [Included / Excluded at Criterion 'i'] </Answer>.           
    """


    template = """
        ## Our systematic review is governed by the following objectives: 
            {objectives}

        ## Provided Criteria:
            
            ### Criterion 1: Randomized Controlled Trial Identification
                Determine whether the study explicitly describes or implicitly implies a **randomized controlled trial/randomized clinical trial**.
                Think step by step and return the detailed reasoning process. Return:
                - Reasons: <Reasons> [reasoning process] </Reasons>

                1. **If the study explicitly describes or implicitly implies it is a randomized controlled trial/randomized clinical trial**:
                - Return: <Ambiguity> None </Ambiguity>
                - Execute: Proceed to the next Criterion

                2. **If there is insufficient information to decide whether the study explicitly describes or implicitly implies it is a randomized controlled trial/randomized clinical trial**:
                - Return: <Ambiguity> Ambiguity at Criterion 6 </Ambiguity>
                - Execute: Proceed to the next Criterion

                3. **If the study does *not* explicitly describe or implicitly implies it is a randomized controlled trial/randomized clinical trial**:
                - Return: <Answer> Excluded at Criterion 6 </Answer>
                - Execute: Stop here and generate the final Answer

            {inclusion_criteria}

            {exclusion_criteria}

            
        ## Based on the above reasoning, the final Answer is: <Answer> [Included / Excluded at Criterion 'i'] </Answer>
            
    """

    fix_criteria_num = 1
    objectives = data_dict.get('Objective', 'No objective provided.')
    inclusion_criteria = data_dict.get("Inclusion_Criteria", [])
    exclusion_criteria = data_dict.get("Exclusion_Criteria", [])

    inclusion_types = data_dict.get("Inclusion_Type", [])
    exclusion_types = data_dict.get("Exclusion_Type", [])
    assert len(inclusion_criteria)==len(inclusion_types)
    assert len(exclusion_criteria)==len(exclusion_types)

    # 填充Inclusion Criteria
    inclusion_list = []
    last_inclusion_index = fix_criteria_num + len(inclusion_criteria)
    for i, (inclusion, inclusion_type) in enumerate(zip(inclusion_criteria, inclusion_types), start=fix_criteria_num+1):
        
        inclusion_type = "Intervention or Control" if (inclusion_type=="Intervention" or inclusion_type=="Control") else inclusion_type

        if  i == last_inclusion_index and len(exclusion_criteria)==0:
            rule_if_yes = \
            f"""1. **If the criterion is explicitly described or implicitly implied in the study**:
                - Return:  <Answer> Included </Answer>
                - Execute: Stop here and generate the final Answer"""
            rule_if_insufficient = \
            f"""2. **If there is insufficient information to decide** whether the study explicitly describes or implicitly implies that '{inclusion}':
                - Return: <Ambiguity> Ambiguity at Criterion {i} </Ambiguity>  <Answer> Included </Answer>
                - Execute: Stop here and generate the final Answer"""
        else:
            rule_if_yes = \
            f"""1. **If the criterion is explicitly described or implicitly implied in the study**:
                - Return: <Ambiguity> None </Ambiguity>
                - Execute: Proceed to the next Criterion"""
            
            rule_if_insufficient = \
            f"""2. **If there is insufficient information to decide** whether the study explicitly describes or implicitly implies that '{inclusion}':
                - Return: <Ambiguity> Ambiguity at Criterion {i} </Ambiguity>
                - Execute: Proceed to the next Criterion"""

        inclusion_list.append(f"""
            ### Criterion {i}: {inclusion_type} Inclusion
                Determine whether the study explicitly describes or implicitly implies that '{inclusion}'.
                Think step by step and return the detailed reasoning process. When there is still insufficient information after considering synonyms, including semantically equivalent terms, and contextual clues, explain what information is missing. Return:
                - Reasons: <Reasons> [reasoning process] </Reasons>

                {rule_if_yes}

                {rule_if_insufficient}

                3. **If the criterion is *not* explicitly described or implicitly implied in the study**:
                - Return: <Answer> Excluded at Criterion {i} </Answer>
                - Execute: Stop here and generate the final Answer
                """)
    inclusion_criteria_filled = "\n".join(inclusion_list) if inclusion_list else ""

    # 填充Exclusion Criteria
    exclusion_list = []
    start_index = fix_criteria_num + len(inclusion_criteria) 
    last_exclusion_index = start_index + len(exclusion_criteria)  if exclusion_criteria else start_index
    for i, (exclusion, exclusion_type) in enumerate(zip(exclusion_criteria,exclusion_types), start=start_index+1):
        
        exclusion_type = "Intervention or Control" if (exclusion_type=="Intervention" or exclusion_type=="Control") else exclusion_type

        if i == last_exclusion_index:
            rule_if_not = \
            f"""1. **If the criterion is *not* explicitly described in the study**:
                - Return: <Answer> Included </Answer>
                - Execute: Stop here and generate the final Answer"""
            rule_if_insufficient = \
            f"""2. **If there is insufficient information to decide** whether the study explicitly describes that '{exclusion}':
                - Return: <Ambiguity> Ambiguity at Criterion {i} </Ambiguity> <Answer> Included </Answer>
                - Execute: Stop here and generate the final Answer"""
        
        else:
            rule_if_not = \
            f"""1. **If the criterion is *not* explicitly described in the study**:
                - Return: <Ambiguity> None </Ambiguity>
                - Execute: Proceed to the next Criterion"""
            rule_if_insufficient = \
            f"""2. **If there is insufficient information to decide** whether the study explicitly describes that '{exclusion}':
                - Return: <Ambiguity> Ambiguity at Criterion {i} </Ambiguity>
                - Execute: Proceed to the next Criterion"""
        
        exclusion_list.append(f"""
            ### Criterion {i}: {exclusion_type} Exclusion
                Determine whether the study explicitly describes that '{exclusion}'.
                Think step by step and return the detailed reasoning process. When there is still insufficient information after considering synonyms, including semantically equivalent terms, and contextual clues, explain what information is missing. Return:
                - Reasons: <Reasons> [reasoning process] </Reasons>

                {rule_if_not}

                {rule_if_insufficient}

                3. **If the criterion is explicitly described in the study**:
                - Return: <Answer> Excluded at Criterion {i} </Answer>
                - Execute: Stop here and generate the final Answer
                """)
        
    exclusion_criteria_filled = "\n".join(exclusion_list) if exclusion_list else ""

    final_template = template.format(
        objectives=objectives,
        inclusion_criteria=inclusion_criteria_filled,
        exclusion_criteria=exclusion_criteria_filled
    )
 
    return system_prompt, final_template, last_exclusion_index 



def return_messages(df_filtered, record):
    system_prompt, screening_criteria, last_exclusion_index = fill_template(df_filtered)
    record_info = f"Title: {record.get('Title', '')}\nAbstract: {record.get('Abstract', '')}\n"

    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": f"{screening_criteria}\n\nHere is one record:\n{record_info}\n"
        }
    ]

    return messages, screening_criteria, last_exclusion_index


