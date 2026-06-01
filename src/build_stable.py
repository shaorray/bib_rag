#!/usr/bin/env python3
"""
bib_rag stable build - CPU embedding with checkpoint resume
Slow but reliable. Supports interrupt/resume.
"""

import os, sys, re, json, hashlib, time
from pathlib import Path
from datetime import datetime

from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import Chroma

KB_ROOT = "/Disk_bot/Eph/bib_rag"
CHROMA_DB_PATH = f"{KB_ROOT}/chroma_db_new"
METADATA_LOG = f"{KB_ROOT}/data/incremental_metadata.json"
CHECKPOINT_FILE = f"{KB_ROOT}/data/build_checkpoint.json"
BGE_M3_PATH = "/Disk_bot/models/bge-m3"

# ============== 文本处理 ==============

def clean_text(text):
    lines = text.split('\n')
    cleaned = []
    for line in lines:
        if '==> image' in line.lower() or '==> picture' in line.lower(): continue
        if line.strip().startswith('©') and 'rights reserved' in line: continue
        if 'As a library, NLM provides access' in line: continue
        cleaned.append(line)
    return '\n'.join(cleaned)

def truncate_at_references(text):
    lines = text.split('\n'); cutoff = len(lines)
    for i, line in enumerate(lines):
        if re.match(r'^#*\s*(REFERENCES?|BIBLIOGRAPHY|ACKNOWLEDGMENTS?|SUPPLEMENTARY|APPENDIX|DATA AVAILABILITY|CONFLICT OF INTEREST|AUTHOR CONTRIBUTIONS?)\s*$', line, re.I):
            cutoff = i; break
    return '\n'.join(lines[:cutoff])

def extract_sections(text):
    sections = {}; lines = text.split('\n')
    current = None; content = []
    patterns = {
        'abstract': r'^#*\s*(ABSTRACT|SUMMARY)\s*$',
        'introduction': r'^#*\s*(INTRODUCTION|BACKGROUND)\s*$',
        'methods': r'^#*\s*(METHODS?|MATERIALS?|EXPERIMENTAL)\s*',
        'results': r'^#*\s*(RESULTS?|FINDINGS?)\s*$',
        'discussion': r'^#*\s*DISCUSSION\s*$',
        'conclusion': r'^#*\s*CONCLUSIONS?\s*$',
    }
    for line in lines:
        matched = False
        for sec_name, pat in patterns.items():
            if re.match(pat, line, re.I):
                if current and content: sections[current] = '\n'.join(content).strip()
                current = sec_name; content = []; matched = True; break
        if not matched and current: content.append(line)
    if current and content: sections[current] = '\n'.join(content).strip()
    return sections

def smart_chunk(text, source, section, chunk_size=800, overlap=100, max_size=1200):
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    chunks = []; current = []; words = 0; idx = 0
    for para in paragraphs:
        pw = len(para.split())
        if current and (words + pw > max_size):
            ct = '\n\n'.join(current)
            chunks.append({'text': ct, 'source': source, 'section': section, 'idx': idx, 'wc': len(ct.split())})
            current = (current[-2:] if len(current) >= 2 else [current[-1]])
            words = sum(len(p.split()) for p in current); idx += 1
        current.append(para); words += pw
        if words >= chunk_size and words >= max_size * 0.8:
            ct = '\n\n'.join(current)
            chunks.append({'text': ct, 'source': source, 'section': section, 'idx': idx, 'wc': len(ct.split())})
            current = []; words = 0; idx += 1
    if current:
        ct = '\n\n'.join(current)
        chunks.append({'text': ct, 'source': source, 'section': section, 'idx': idx, 'wc': len(ct.split())})
    return chunks

def extract_meta(text, filename):
    meta = {'title': '', 'authors': '', 'year': '', 'journal': '', 'doi': '', 'pmid': '', 'pmcid': ''}
    m = re.search(r'PMID:\s*(\d+)', text)
    if m: meta['pmid'] = m.group(1)
    m = re.search(r'PMCID:\s*(PMC\d+)', text)
    if m: meta['pmcid'] = m.group(1)
    m = re.search(r'(?:doi:|DOI:|https?://doi\.org/)(10\.\S+)', text)
    if m: meta['doi'] = m.group(1)
    m = re.search(r'\b(19\d{2}|20\d{2})\b', text)
    if m: meta['year'] = m.group(1)
    lines = [l.strip() for l in text.split('\n') if l.strip()]
    if lines and 20 < len(lines[0]) < 300: meta['title'] = lines[0]
    name = re.sub(r'^[\+\^\s]+', '', Path(filename).stem)
    if len(name) > 10 and (not meta['title'] or len(name) > len(meta['title'])): meta['title'] = name
    m = re.search(r'^([A-Z][A-Za-z\s\&\.]+)\s*\.?\s*\d{4}', text, re.M)
    if m: meta['journal'] = m.group(1).strip()
    return meta

# ============== 主流程 ==============

def build_rag(papers_dir, batch_size=50):
    md_files = sorted(Path(papers_dir).rglob('*.md'))
    total = len(md_files)
    print(f"\n{'='*60}")
    print(f"📚 bib_rag 知识库构建 (stable CPU)")
    print(f"   论文: {total} 篇")
    print(f"   批次: {batch_size} 篇/批")
    print(f"{'='*60}\n")
    
    # Load checkpoint
    checkpoint = {'processed': set()}
    if os.path.exists(CHECKPOINT_FILE):
        try:
            with open(CHECKPOINT_FILE, 'r') as f:
                cp = json.load(f)
                checkpoint['processed'] = set(cp.get('processed', []))
            print(f"📋 检查点: 已处理 {len(checkpoint['processed'])} 篇")
        except: pass
    
    # Setup paths
    os.makedirs(CHROMA_DB_PATH, exist_ok=True)
    os.makedirs(os.path.dirname(METADATA_LOG), exist_ok=True)
    
    # Load embedding model
    print("🔌 Loading bge-m3 (CPU)...")
    model = SentenceTransformer(BGE_M3_PATH, trust_remote_code=True, device='cpu')
    print(f"   ✅ dim={model.get_sentence_embedding_dimension()}")
    
    # Connect Chroma
    class Dummy:
        def embed_documents(self, texts): return [[0.0]*1024 for _ in texts]
        def embed_query(self, text): return [0.0]*1024
    db = Chroma(persist_directory=CHROMA_DB_PATH, embedding_function=Dummy(), collection_name="bib_rag_papers")
    
    # Load metadata
    metadata = {}
    if os.path.exists(METADATA_LOG):
        try:
            with open(METADATA_LOG, 'r') as f: metadata = json.load(f)
        except: pass
    
    stats = {'success': 0, 'failed': 0, 'skipped': 0, 'chunks': 0}
    start_time = time.time()
    
    for batch_start in range(0, total, batch_size):
        files = md_files[batch_start:batch_start + batch_size]
        bn = batch_start // batch_size + 1
        tbn = (total - 1) // batch_size + 1
        
        docs = []; texts_to_embed = []; metas = {}; count = 0
        for i, fp in enumerate(files, batch_start + 1):
            # Skip already processed
            if fp.name in checkpoint['processed']:
                stats['skipped'] += 1
                continue
            
            try:
                text = fp.read_text(encoding='utf-8', errors='ignore')
                if len(text.strip()) < 500: 
                    checkpoint['processed'].add(fp.name)
                    stats['skipped'] += 1; continue
                
                cleaned = clean_text(text)
                cleaned = truncate_at_references(cleaned)
                if len(cleaned.strip()) < 500:
                    checkpoint['processed'].add(fp.name)
                    stats['skipped'] += 1; continue
                
                ch = hashlib.md5(cleaned.encode()).hexdigest()[:16]
                if fp.name in metadata and metadata[fp.name].get('hash') == ch:
                    checkpoint['processed'].add(fp.name)
                    stats['skipped'] += 1; continue
                
                meta = extract_meta(cleaned, fp.name)
                secs = extract_sections(cleaned)
                sf = list(secs.keys())
                
                chunks = []
                if secs:
                    for sn, st in secs.items():
                        if st.strip(): chunks.extend(smart_chunk(st, fp.name, sn))
                else:
                    chunks = smart_chunk(cleaned, fp.name, "full_text")
                
                if not chunks:
                    checkpoint['processed'].add(fp.name)
                    stats['skipped'] += 1; continue
                
                for c in chunks:
                    docs.append({
                        'text': c['text'],
                        'meta': {'source': fp.name, 'section': c['section'], 'idx': c['idx'],
                                'wc': c['wc'], 'title': meta.get('title', fp.name),
                                'authors': meta.get('authors', ''), 'year': meta.get('year', ''),
                                'journal': meta.get('journal', ''), 'doi': meta.get('doi', ''),
                                'pmid': meta.get('pmid', ''), 'pmcid': meta.get('pmcid', ''),
                                'hash': ch}
                    })
                    texts_to_embed.append(c['text'])
                
                metas[fp.name] = {
                    'title': meta.get('title', fp.name), 'year': meta.get('year', ''),
                    'chunks': len(chunks), 'hash': ch, 'sections': sf,
                    'added': datetime.now().isoformat()
                }
                count += 1
                
            except Exception as e:
                stats['failed'] += 1
                print(f"  [{i}] ❌ {fp.name}: {str(e)[:80]}")
        
        if not docs:
            # Save checkpoint even if no docs
            with open(CHECKPOINT_FILE, 'w') as f:
                json.dump({'processed': sorted(checkpoint['processed'])}, f)
            continue
        
        print(f"\n📦 批次 {bn}/{tbn}: {count} 篇, {len(texts_to_embed)} chunks")
        
        # Embed in sub-batches
        print(f"   🧠 Embedding (CPU)...")
        t0 = time.time()
        embeddings = []
        sub_batch = 16
        for j in range(0, len(texts_to_embed), sub_batch):
            batch = texts_to_embed[j:j+sub_batch]
            embs = model.encode(batch, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
            embeddings.extend(embs.tolist())
            if (j // sub_batch + 1) % 10 == 0:
                print(f"      {j+len(batch)}/{len(texts_to_embed)} done")
        embed_time = time.time() - t0
        print(f"   ✅ Embedded in {embed_time:.1f}s ({len(texts_to_embed)/embed_time:.1f} chunks/s)")
        
        # Write to Chroma
        print(f"   💾 Writing to ChromaDB...")
        t0 = time.time()
        try:
            ids = [f"{d['meta']['source']}_{d['meta']['idx']}" for d in docs]
            db._collection.add(
                ids=ids,
                documents=[d['text'] for d in docs],
                metadatas=[d['meta'] for d in docs],
                embeddings=embeddings
            )
            print(f"   ✅ Written in {time.time()-t0:.1f}s")
            
            # Update metadata and checkpoint
            metadata.update(metas)
            for fname in metas:
                checkpoint['processed'].add(fname)
            
            with open(METADATA_LOG, 'w') as f: json.dump(metadata, f, ensure_ascii=False, indent=2)
            with open(CHECKPOINT_FILE, 'w') as f:
                json.dump({'processed': sorted(checkpoint['processed'])}, f)
            
            stats['success'] += count; stats['chunks'] += len(docs)
        except Exception as e:
            print(f"   ❌ Failed: {e}")
            stats['failed'] += count
        
        elapsed = time.time() - start_time
        rate = stats['success'] / elapsed * 60 if elapsed > 0 else 0
        remaining = (total - stats['success'] - stats['skipped']) / (stats['success'] / elapsed) if stats['success'] > 0 else 0
        print(f"\n📊 进度: {stats['success']}/{total} 成功, {stats['skipped']} 跳过, {stats['failed']} 失败")
        print(f"   速度: {rate:.1f} 篇/分钟 | 预计剩余: {remaining/60:.1f} 小时")
    
    print(f"\n{'='*60}")
    print(f"✅ 完成!")
    print(f"   总计: {total}")
    print(f"   成功: {stats['success']} ({stats['chunks']} chunks)")
    print(f"   跳过: {stats['skipped']}")
    print(f"   失败: {stats['failed']}")
    print(f"   耗时: {(time.time()-start_time)/3600:.1f} 小时")
    print(f"{'='*60}")
    return stats

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('papers_dir', help='论文 markdown 目录')
    parser.add_argument('-b', '--batch-size', type=int, default=50)
    args = parser.parse_args()
    
    print("="*60)
    print("📚 bib_rag 知识库构建 (stable CPU)")
    print("="*60)
    
    result = build_rag(args.papers_dir, args.batch_size)
    print(f"\n📊 Final: {result}")
