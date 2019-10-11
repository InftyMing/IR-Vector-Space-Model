import inquire
query_type = input('Please select the type of information you want to query(1-3):\n[1]Authors\n[2]Subject\n[3]Contents\n')

if query_type == '1' or query_type == '2' or query_type == '3':
    query_text = input('Please input the information you want to query: ')
    cosine_score = inquire.Vector_Space_Model(query_text, inquire.read_pkl('doc_num.pkl'), int(query_type))
    if cosine_score is not None:
        inquire.find_doc_path(cosine_score)
else:
    print('Please enter correct serial number!')
