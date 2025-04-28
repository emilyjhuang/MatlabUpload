# MATLAB to Python code


Upload information MATlab to python


**4. Create a Virtual Environment:**

- _Mac:_

  ```
  python3 -m venv .venv
  source .venv/bin/activate
  ```

- _Windows:_
  ```
  python -m venv .venv
  .venv\Scripts\activate
  ```

**5. Install Dependencies:**

```
pip install -r requirements.txt
```


```
uvicorn main:app --reload

```

fresh environment

```
rm -rf .venv
python -m venv .venv
source .venv/bin/activate  # On Mac/Linux
pip install -r requirements.txt

```
