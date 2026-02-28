import os
BASE = os.path.dirname(os.path.abspath(__file__))
for d in ["core","database","api","ui","demo","scripts",
          os.path.join("data","rules"), os.path.join("data","vehicles"),
          os.path.join("data","chroma_db")]:
    os.makedirs(os.path.join(BASE, d), exist_ok=True)
    with open(os.path.join(BASE, d, "__init__.py"), "w") as f:
        pass
print("done")
