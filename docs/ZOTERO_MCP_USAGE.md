# Zotero MCP Usage Guide for OpenClaw

## Overview

Zotero 9 (ESR 140) runs locally with a HTTP API on **port 23119**. Library has ~3,739 items.

- **Base URL**: `http://localhost:23119/api`
- **User ID**: `0` (local user)
- **Profile**: `~/.zotero/zotero/n34q4qw9.default/`
- **Install**: `uv tool install zotero-mcp`

## Key API Endpoints

### List items
```
GET /api/users/0/items/top?limit=10
```
Returns list of items. Each item has `key`, `meta`, and `data` fields. Item details (title, authors, DOI, etc.) are in `item["data"]`.

### Search items
```
GET /api/users/0/items?q=eph%20receptor&limit=5
```
Searches title, abstract, creators, notes.

### Get item by key
```
GET /api/users/0/items/{itemKey}
```
Returns full item with all fields in `item["data"]`:
- `title`, `itemType`, `creators`, `DOI`, `date`, `publicationTitle`
- `volume`, `issue`, `pages`, `abstract`, `url`, `tags`

### List collections
```
GET /api/users/0/collections
```

### Items in a collection
```
GET /api/users/0/collections/{collectionKey}/items?limit=20
```

## Python Example

```python
import requests

BASE = "http://localhost:23119/api/users/0"

# Search
def search(query, limit=10):
    r = requests.get(f"{BASE}/items", params={"q": query, "limit": limit})
    return r.json()

# Get item details
def get_item(item_key):
    r = requests.get(f"{BASE}/items/{item_key}")
    data = r.json()
    # Details are in data["data"]
    return data.get("data", data)

# Search Eph receptors
items = search("eph receptor", limit=5)
for item in items:
    d = item.get("data", item)
    print(f"{d.get('title','')} ({d.get('date','')}) - DOI: {d.get('DOI','')}")
```

## Common Patterns

### Search by DOI
```bash
curl -s "http://localhost:23119/api/users/0/items?q=10.1038%2Fnrc2806&limit=1"
```

### Get item by key
```bash
curl -s "http://localhost:23119/api/users/0/items/QYC4K7BG"
```

### List top-level collections
```bash
curl -s "http://localhost:23119/api/users/0/collections"
```

## Data Structure

```json
{
  "key": "QYC4K7BG",
  "version": 2,
  "data": {
    "itemType": "journalArticle",
    "title": "Paper Title",
    "creators": [{"firstName": "First", "lastName": "Last", "creatorType": "author"}],
    "DOI": "10.xxxx/xxxxx",
    "date": "2024",
    "publicationTitle": "Journal Name",
    "volume": "12",
    "issue": "3",
    "pages": "45-67",
    "tags": [{"tag": "eph", "type": 1}]
  }
}
```

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Connection refused | Zotero must be running; check `zotero -ZoteroDebug` |
| Empty data fields | Details are in `item["data"]`, not top-level |
| 404 | Check item key (20-char alphanumeric) |

## Integration with bib_rag

The `bib_rag` KG-RAG system uses Zotero as a citation lookup backend:
- DOI is the reliable key for KG↔Zotero matching
- `make_zotero_key()` generates human-readable keys from DOI + title + year
- 284/1,104 KG entries are cross-referenced with Zotero
- 820 entries unmatched (not in Zotero library)
