import re
string = '982.58亿元（2018年末）[1]</sup><a class="sup-anchor" name="ref_[1]_440557"> </a>'
print(re.sub('<.*?>', '', string))

