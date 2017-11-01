# calculate_statistics_abnormal
# calculate_abnormal
# calculate_statistics
# calculate_regression
command = 'calculate_statistics'
# average, variance, standard_deviation,
command_detail = 'correlation'

dict = {}
dict['command'] = command
dict['command_to'] = 'server'
time_start = '2017-07-01 00:00:00'
time_end = '2017-08-01 00:00:00'

if command == 'calculate_statistics_abnormal':
    dict['section'] = 'one'
    dict['time_start'] = time_start
    dict['time_end'] = time_end
elif command == 'calculate_abnormal':
    dict['section_one_standard_discharge'] = '1539'
    dict['section_two_standard_discharge'] = '3981'
elif command == 'calculate_statistics':
    if command_detail == 'correlation':
        dict['command_detail'] = 'correlation'
        dict['location_one'] = {'location':'seongsan', 'type':'discharge'}
        dict['location_two'] = {'location':'gimpo', 'type':'pressure'}
        dict['time_start'] = time_start
        dict['time_end'] = time_end
    else:
        dict['command_detail'] = command_detail
        dict['location'] = 'd.seongsan - d.gimpo'
        dict['time_start'] = time_start
        dict['time_end'] = time_end
elif command == 'calculate_regression':
    dict['location_source'] = '[d.yangje - d.gwangam][d.yangje - d.gimpo]'
    dict['location_target'] = 'd.paldang + d.36t'
    dict['time_start'] = time_start
    dict['time_end'] = time_end

print(dict['location_one']['location'])
print(dict['location_one']['type'])
print(dict['location_two']['location'])
print(dict['location_two']['type'])





corr_x = ''
corr_y = ''

if dict['location_one']['type'] == 'discharge':
    corr_x = 'd.' + dict['location_one']['location']
elif dict['location_one']['type'] == 'pressure':
    corr_x = 'p.' + dict['location_one']['location']

if dict['location_two']['type'] == 'discharge':
    corr_y = 'd.' + dict['location_two']['location']
elif dict['location_two']['type'] == 'pressure':
    corr_y = 'p.' + dict['location_two']['location']

print(corr_x)
print(corr_y)