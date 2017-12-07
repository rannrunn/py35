def validate_json(dict):

    result = ''

    if dict['command'] == '':
        result += 'validation error : command , '
    # calculate_regression 빼고 나머지는 command_detail이 존재한다.
    if dict['command'] != 'calculate_regression' and dict['command_detail'] == '':
        result += 'validation error : command_detail , '
    if dict['table'] == '':
        result += 'validation error : table , '
    if dict['input'] == '':
        result += 'validation error : input , '
    if dict['output'] == '':
        result += 'validation error : output , '

    dict['error'] = result[:-3]

    #print('error:', dict['error'])

    return dict

