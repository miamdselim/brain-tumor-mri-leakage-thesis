import json,sys,re
s=open(sys.argv[1]).read()
parts=re.split(r'^# %%(.*)$',s,flags=re.M)
cells=[]
for i in range(1,len(parts),2):
    tag,body=parts[i],parts[i+1].strip('\n')
    if '[markdown]' in tag:
        src='\n'.join(l[2:] if l.startswith('# ') else l.lstrip('#') for l in body.splitlines())
        cells.append({"cell_type":"markdown","metadata":{},"source":src.splitlines(True)})
    else:
        cells.append({"cell_type":"code","metadata":{},"execution_count":None,"outputs":[],"source":body.splitlines(True)})
nb={"cells":cells,"metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},"language_info":{"name":"python"}},"nbformat":4,"nbformat_minor":5}
json.dump(nb,open(sys.argv[2],'w'),indent=1)
print(len(cells),'cells')
