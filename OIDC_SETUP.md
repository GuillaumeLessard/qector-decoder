# PyPI OIDC Trusted Publisher Setup

Complete these steps **once** in a browser before pushing the first release tag.

---

## Step 1: PyPI Trusted Publisher (2 minutes)

1. Go to <https://pypi.org/manage/account/publishing/>
2. Log in with the `qectorlab` / `admin@qector.store` account
3. Under **"Add a new pending publisher"**, fill in:

   | Field | Value |
   |-------|-------|
   | PyPI Project Name | `qector-decoder-v3` |
   | Owner | `qectorlab` |
   | Repository name | `qector-decoder` |
   | Workflow filename | `CI.yml` |
   | Environment name | `pypi` |

4. Click **Add**.

---

## Step 2: GitHub Environment (1 minute)

1. Go to <https://github.com/GuillaumeLessard/qector-decoder/settings/environments>
2. Click **New environment**
3. Name it exactly: `pypi`
4. Click **Configure environment**
5. Under **Deployment branches and tags**, select **Selected branches and tags**
6. Add rule: `refs/tags/v*`
7. Save.

---

## Step 3: Add GitHub Secrets (already done via API)

The Rust source archive was split across **three** secrets and uploaded automatically.
To verify they are present, go to:
<https://github.com/GuillaumeLessard/qector-decoder/settings/secrets/actions>

You should see: `RUST_SRC_B64_1`, `RUST_SRC_B64_2`, `RUST_SRC_B64_3`

---

## Step 4: Regenerate the secret archive (when source changes)

```powershell
# Run from C:\Users\Admin\Desktop\qector-decoder-main
tar -czf C:\Users\Admin\Desktop\qector_src_only.tar.gz `
    -C C:\Users\Admin\Desktop\qector-decoder-main src build.rs proto

$b64 = [Convert]::ToBase64String([IO.File]::ReadAllBytes("C:\Users\Admin\Desktop\qector_src_only.tar.gz"))
$third = [int]($b64.Length / 3)
[IO.File]::WriteAllText("C:\Users\Admin\Desktop\src_p1.txt", $b64.Substring(0, $third))
[IO.File]::WriteAllText("C:\Users\Admin\Desktop\src_p2.txt", $b64.Substring($third, $third))
[IO.File]::WriteAllText("C:\Users\Admin\Desktop\src_p3.txt", $b64.Substring($third * 2))

# Then re-run the upload script:
python C:\Users\Admin\Desktop\upload_secrets.py <GITHUB_TOKEN> <KEY_ID> <PUB_KEY>
```

---

## Step 5: Trigger a release

```powershell
cd C:\Users\Admin\Desktop\qector-push-clean
git tag v0.5.0
git push origin v0.5.0
```

CI will build wheels for all platforms and publish to PyPI automatically.
