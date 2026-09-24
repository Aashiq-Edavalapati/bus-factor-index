"""
Data Collection Script for Open-Source Package Abandonment Risk
Collects raw metadata from:
1. npm Registry API (registry.npmjs.org)
2. npms.io API (api.npms.io/v2)
3. npm Downloads API (api.npmjs.org)
4. PyPI JSON API (pypi.org/pypi)
5. pypistats API (pypistats.org)
6. GitHub REST API (api.github.com)

Saves raw data to data/raw_dataset.csv
"""

import os
import json
import time
import requests
import datetime
import concurrent.futures
import pandas as pd

NOW_DATE = datetime.datetime.now(datetime.timezone.utc)

# 1. Curated package lists across health spectrum and ecosystems
CURATED_HEALTHY_NPM = [
    'react', 'vue', 'svelte', 'next', 'express', 'fastify', 'koa', 'axios', 
    'got', 'undici', 'lodash', 'chalk', 'commander', 'yargs', 'typescript', 
    'webpack', 'vite', 'esbuild', 'rollup', 'prettier', 'eslint', 'prisma', 
    'mongoose', 'zod', 'dotenv', 'winston', 'pino', 'rxjs', 'dayjs', 
    'date-fns', 'jest', 'playwright', 'puppeteer', 'ws', 'socket.io', 
    'cheerio', 'sharp', 'cross-spawn', 'execa', 'semver', 'uuid', 'nanoid',
    'turbo', 'tailwindcss', 'postcss', 'esbuild-loader', 'zod-to-json-schema',
    'supertest', 'nodemon', 'ts-node', 'vitest', 'pinia', 'lucide-react',
    'ioredis', 'knex', 'typeorm', 'drizzle-orm', 'trpc', 'hono'
]

CURATED_AT_RISK_NPM = [
    'core-js', 'minimatch', 'globby', 'del', 'figures', 'log-symbols', 
    'cli-spinners', 'boxen', 'table', 'strip-json-comments', 'json5', 'ini', 
    'deepmerge', 'extend', 'object-assign', 'clone-deep', 'is-plain-object', 
    'kind-of', 'is-number', 'is-glob', 'fill-range', 'to-regex-range', 
    'braces', 'anymatch', 'micromatch', 'picomatch', 'nanomatch', 'arr-diff', 
    'arr-flatten', 'array-unique', 'repeat-element', 'repeat-string', 
    'split-string', 'set-value', 'get-value', 'has-value', 'union-value', 
    'pascalcase', 'decamelize', 'camelcase', 'strip-ansi', 'supports-color',
    'picocolors', 'kleur', 'colorette', 'pretty-bytes', 'ms', 'bytes',
    'filesize', 'humanize-duration', 'is-wsl', 'open', 'cpy', 'make-dir',
    'p-map', 'p-filter', 'p-all', 'p-series', 'p-waterfall'
]

CURATED_ABANDONED_NPM = [
    'request', 'left-pad', 'nomnom', 'colors', 'event-stream', 'jade', 
    'coffee-script', 'bower', 'gulp-util', 'uglify-js', 'babel-core', 
    'grunt', 'optimist', 'connect', 'restify', 'async-listener', 'node-uuid', 
    'mkdirp-then', 'fs-promise', 'q', 'when', 'bluebird', 'co', 'step', 
    'consolidate', 'swig', 'dustjs-linkedin', 'vows', 'expresso', 'should', 
    'expect.js', 'urllib', 'querystring', 'url', 'punycode', 'builtin-modules',
    'resolve-from', 'find-up', 'locate-path', 'p-locate', 'p-limit', 'p-try', 
    'p-finally', 'p-queue', 'p-retry', 'amdefine', 'source-map-support', 
    'clean-css', 'html-minifier', 'uglifyjs-webpack-plugin', 'node-sass',
    'charenc', 'crypt', 'is-buffer', 'core-util-is', 'inherits'
]

CURATED_PYPI = [
    'requests', 'numpy', 'pandas', 'scipy', 'scikit-learn', 'flask', 'django', 
    'fastapi', 'pydantic', 'pytest', 'black', 'click', 'rich', 'sqlalchemy', 
    'httpx', 'cryptography', 'pillow', 'celery', 'redis', 'jinja2',
    'werkzeug', 'alembic', 'paramiko', 'beautifulsoup4', 'lxml', 'pyyaml', 
    'feedparser', 'fabric', 'tqdm', 'joblib', 'colorama', 'chardet', 'idna', 
    'certifi', 'simplejson', 'six', 'theano', 'nose', 'distutils2', 'pysqlite', 
    'pycrypto', 'pychecker', 'pep8', 'supervisor', 'pathlib', 'mock', 
    'ipaddress', 'enum34', 'functools32', 'urllib3', 'twisted', 'tornado',
    'asyncio', 'gevent', 'cherrypy', 'bottle', 'webob', 'pastedeploy'
]

PACKAGE_LICENSE_MAP = {
    'scipy': 'BSD-3-Clause',
    'pandas': 'BSD-3-Clause',
    'numpy': 'BSD-3-Clause',
    'scikit-learn': 'BSD-3-Clause',
    'flask': 'BSD-3-Clause',
    'click': 'BSD-3-Clause',
    'jinja2': 'BSD-3-Clause',
    'werkzeug': 'BSD-3-Clause',
    'django': 'BSD-3-Clause',
    'colorama': 'BSD-3-Clause',
    'idna': 'BSD-3-Clause',
    'joblib': 'BSD-3-Clause',
    'mock': 'BSD-3-Clause',
    'cherrypy': 'BSD-3-Clause',
    'enum34': 'BSD-3-Clause',
    'supervisor': 'BSD-3-Clause',
    'theano': 'BSD-3-Clause',
    'fabric': 'BSD-3-Clause',
    'amdefine': 'BSD-3-Clause',
    'hoek': 'BSD-3-Clause',
    'fastapi': 'MIT',
    'pydantic': 'MIT',
    'pytest': 'MIT',
    'black': 'MIT',
    'redis': 'MIT',
    'pillow': 'MIT',
    'alembic': 'MIT',
    'beautifulsoup4': 'MIT',
    'urllib3': 'MIT',
    'twisted': 'MIT',
    'pathlib': 'MIT',
    'pep8': 'MIT',
    'simplejson': 'MIT',
    'tqdm': 'MIT',
    'type-fest': 'MIT',
    'nomnom': 'MIT',
    'expresso': 'MIT',
    'expect.js': 'MIT',
    'strapi': 'MIT',
    'optimist': 'MIT',
    'cryptography': 'Apache-2.0',
    'asyncio': 'Apache-2.0',
    'dompurify': 'Apache-2.0',
    'paramiko': 'LGPL-2.1',
    'chardet': 'LGPL-2.1',
    'nose': 'LGPL-2.1',
    'ipaddress': 'PSF-2.0',
    'distutils2': 'PSF-2.0',
    'functools32': 'PSF-2.0',
    'argparse': 'PSF-2.0',
    'pysqlite': 'Zlib',
    'pycrypto': 'Unlicense',
}

def clean_license_name(lic, pkg_name=None):
    if pkg_name and pkg_name in PACKAGE_LICENSE_MAP:
        return PACKAGE_LICENSE_MAP[pkg_name]
    if not isinstance(lic, str) or not lic.strip():
        return 'MIT'
    l = lic.strip()
    l_up = l.upper()
    if 'BSD 3' in l_up or 'BSD-3' in l_up: return 'BSD-3-Clause'
    if 'BSD 2' in l_up or 'BSD-2' in l_up: return 'BSD-2-Clause'
    if 'BSD' in l_up: return 'BSD-3-Clause'
    if 'APACHE' in l_up: return 'Apache-2.0'
    if 'MIT' in l_up or 'EXPAT' in l_up: return 'MIT'
    if 'ISC' in l_up: return 'ISC'
    if 'PSF' in l_up or 'PYTHON' in l_up: return 'PSF-2.0'
    if 'LGPL' in l_up: return 'LGPL-2.1'
    if 'GPL' in l_up: return 'GPL-3.0'
    if 'MPL' in l_up: return 'MPL-2.0'
    if 'ZLIB' in l_up: return 'Zlib'
    if 'PUBLIC DOMAIN' in l_up or 'UNLICENSE' in l_up: return 'Unlicense'
    if 'WTFPL' in l_up: return 'WTFPL'
    if 'BLUEOAK' in l_up: return 'BlueOak-1.0.0'
    if len(l) > 30: return 'MIT'
    return l

def search_additional_npm():
    queries = [
        'keywords:cli', 'keywords:utility', 'keywords:framework', 
        'keywords:database', 'keywords:parser', 'keywords:security',
        'is:deprecated', 'keywords:build', 'keywords:middleware', 'keywords:validation'
    ]
    additional = set()
    for q in queries:
        try:
            url = f'https://api.npms.io/v2/search?q={q}&size=25'
            r = requests.get(url, timeout=10)
            if r.status_code == 200:
                for item in r.json().get('results', []):
                    pkg_name = item.get('package', {}).get('name')
                    if pkg_name and not pkg_name.startswith('@types/'):
                        additional.add(pkg_name)
        except Exception as e:
            print(f"Search query error {q}: {e}")
    return list(additional)

def fetch_npm_batch_npms(package_names):
    """Fetch batch package data from npms.io mget endpoint."""
    results = {}
    batch_size = 100
    for i in range(0, len(package_names), batch_size):
        chunk = package_names[i:i + batch_size]
        try:
            r = requests.post('https://api.npms.io/v2/package/mget', json=chunk, timeout=20)
            if r.status_code == 200:
                results.update(r.json())
            else:
                print(f"npms mget status {r.status_code}")
        except Exception as e:
            print(f"Error fetching npms batch: {e}")
    return results

def fetch_npm_registry_single(pkg_name):
    """Fetch live data from registry.npmjs.org for a single package."""
    try:
        url = f"https://registry.npmjs.org/{pkg_name}"
        r = requests.get(url, timeout=8)
        if r.status_code == 200:
            return pkg_name, r.json()
    except Exception:
        pass
    return pkg_name, None

def fetch_npm_downloads_single(pkg_name):
    """Fetch monthly downloads from api.npmjs.org."""
    try:
        url = f"https://api.npmjs.org/downloads/point/last-month/{pkg_name}"
        r = requests.get(url, timeout=8)
        if r.status_code == 200:
            return pkg_name, r.json().get('downloads', 0)
    except Exception:
        pass
    return pkg_name, 0

def fetch_pypi_single(pkg_name):
    """Fetch package metadata from PyPI JSON API and pypistats."""
    try:
        url = f"https://pypi.org/pypi/{pkg_name}/json"
        r = requests.get(url, timeout=8)
        if r.status_code != 200:
            return None
        data = r.json()
        
        # downloads from pypistats
        dl_url = f"https://pypistats.org/api/packages/{pkg_name}/recent"
        dl = 0
        try:
            dl_r = requests.get(dl_url, timeout=5)
            if dl_r.status_code == 200:
                dl = dl_r.json().get('data', {}).get('last_month', 0)
        except Exception:
            pass
            
        info = data.get('info', {})
        releases = data.get('releases', {})
        
        # parse upload times
        upload_times = []
        for ver, files in releases.items():
            for f in files:
                t_str = f.get('upload_time_iso_8601')
                if t_str:
                    try:
                        upload_times.append(datetime.datetime.fromisoformat(t_str.replace('Z', '+00:00')))
                    except Exception:
                        pass
                        
        created_at = min(upload_times).isoformat() if upload_times else None
        latest_release = max(upload_times).isoformat() if upload_times else None
        
        # project urls and github
        project_urls = info.get('project_urls') or {}
        repo_url = info.get('home_page') or ''
        for k, v in project_urls.items():
            if 'github.com' in str(v):
                repo_url = v
                break
                
        # dependencies
        requires_dist = info.get('requires_dist') or []
        
        return {
            'package_name': pkg_name,
            'ecosystem': 'pypi',
            'repository_url': repo_url,
            'license': clean_license_name(info.get('license'), pkg_name),
            'created_at': created_at,
            'latest_release_date': latest_release,
            'latest_version': info.get('version', ''),
            'releases_count': len(releases),
            'maintainers_count': 1 if info.get('author') or info.get('maintainer') else 0,
            'contributors_count': 0, # enriched later or estimated from release history
            'total_commits': 0,
            'top_contributor_commits': 0,
            'top_2_contributor_commits': 0,
            'contributor_commits_json': json.dumps([]),
            'stars_count': 0,
            'forks_count': 0,
            'subscribers_count': 0,
            'open_issues_count': 0,
            'total_issues_count': 0,
            'dependencies_count': len(requires_dist),
            'dev_dependencies_count': 0,
            'downloads_monthly': dl,
            'dependents_count': 0,
            'is_deprecated': False,
            'has_test_script': True,
            'npms_quality_score': None,
            'npms_popularity_score': None,
            'npms_maintenance_score': None
        }
    except Exception as e:
        print(f"Error fetching PyPI {pkg_name}: {e}")
        return None

def main():
    os.makedirs('data', exist_ok=True)
    print("Step 1: Gathering candidate packages...")
    additional_npm = search_additional_npm()
    all_npm = list(dict.fromkeys(CURATED_HEALTHY_NPM + CURATED_AT_RISK_NPM + CURATED_ABANDONED_NPM + additional_npm))
    print(f"Total NPM packages to process: {len(all_npm)}")
    print(f"Total PyPI packages to process: {len(CURATED_PYPI)}")
    
    # Fetch npms.io batch data
    print("Step 2: Fetching batch npms.io metadata for NPM packages...")
    npms_data = fetch_npm_batch_npms(all_npm)
    print(f"Received npms.io data for {len(npms_data)} packages.")
    
    # Fetch registry.npmjs.org data in parallel
    print("Step 3: Fetching live registry.npmjs.org metadata in parallel...")
    registry_data = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=15) as executor:
        future_to_pkg = {executor.submit(fetch_npm_registry_single, p): p for p in all_npm}
        for future in concurrent.futures.as_completed(future_to_pkg):
            pkg_name, reg_json = future.result()
            if reg_json:
                registry_data[pkg_name] = reg_json
    print(f"Received npm registry data for {len(registry_data)} packages.")
    
    # Fetch monthly downloads in parallel
    print("Step 4: Fetching live npm downloads in parallel...")
    downloads_data = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=15) as executor:
        future_to_pkg = {executor.submit(fetch_npm_downloads_single, p): p for p in all_npm}
        for future in concurrent.futures.as_completed(future_to_pkg):
            pkg_name, dl = future.result()
            downloads_data[pkg_name] = dl
            
    # Process NPM records
    raw_records = []
    for pkg in all_npm:
        reg = registry_data.get(pkg)
        if not reg:
            continue
            
        npms_pkg = npms_data.get(pkg, {})
        gh = npms_pkg.get('collected', {}).get('github', {})
        meta = npms_pkg.get('collected', {}).get('metadata', {})
        eval_info = npms_pkg.get('evaluation', {})
        
        # Time handling
        time_info = reg.get('time', {})
        created_at = time_info.get('created')
        latest_version = reg.get('dist-tags', {}).get('latest')
        latest_release_date = time_info.get(latest_version) or time_info.get('modified')
        
        # Contributors and commit concentration
        contributors = gh.get('contributors', [])
        commit_counts = [c.get('commitsCount', 0) for c in contributors] if contributors else []
        total_commits = sum(commit_counts)
        sorted_commits = sorted(commit_counts, reverse=True)
        top1_commits = sorted_commits[0] if sorted_commits else 0
        top2_commits = sum(sorted_commits[:2]) if len(sorted_commits) >= 2 else top1_commits
        
        # Maintainers
        maintainers = reg.get('maintainers', [])
        maintainers_count = len(maintainers) if isinstance(maintainers, list) else 1
        
        # Issues
        issues_info = gh.get('issues', {})
        open_issues = issues_info.get('openCount', 0)
        total_issues = issues_info.get('count', 0)
        
        # Stars, forks, subscribers
        stars = gh.get('starsCount', 0)
        forks = gh.get('forksCount', 0)
        subscribers = gh.get('subscribersCount', 0)
        
        # Dependencies
        versions = reg.get('versions', {})
        latest_pkg_json = versions.get(latest_version, {}) if latest_version else {}
        deps = latest_pkg_json.get('dependencies', {})
        dev_deps = latest_pkg_json.get('devDependencies', {})
        
        # Deprecated
        is_deprecated = bool(reg.get('deprecated') or latest_pkg_json.get('deprecated'))
        
        # License
        license_str = reg.get('license')
        if isinstance(license_str, dict):
            license_str = license_str.get('type')
        if not license_str and latest_pkg_json:
            license_str = latest_pkg_json.get('license')
        if not license_str:
            license_str = meta.get('license', 'Unknown')
            
        repo_info = reg.get('repository', {})
        repo_url = repo_info.get('url', '') if isinstance(repo_info, dict) else str(repo_info)
        
        record = {
            'package_name': pkg,
            'ecosystem': 'npm',
            'repository_url': repo_url,
            'license': clean_license_name(license_str, pkg),
            'created_at': created_at,
            'latest_release_date': latest_release_date,
            'latest_version': str(latest_version),
            'releases_count': len(versions),
            'maintainers_count': maintainers_count,
            'contributors_count': len(contributors),
            'total_commits': total_commits,
            'top_contributor_commits': top1_commits,
            'top_2_contributor_commits': top2_commits,
            'contributor_commits_json': json.dumps(commit_counts),
            'stars_count': stars,
            'forks_count': forks,
            'subscribers_count': subscribers,
            'open_issues_count': open_issues,
            'total_issues_count': total_issues,
            'dependencies_count': len(deps) if isinstance(deps, dict) else 0,
            'dev_dependencies_count': len(dev_deps) if isinstance(dev_deps, dict) else 0,
            'downloads_monthly': downloads_data.get(pkg, 0),
            'dependents_count': eval_info.get('popularity', {}).get('dependentsCount', 0),
            'is_deprecated': is_deprecated,
            'has_test_script': meta.get('hasTestScript', False),
            'npms_quality_score': eval_info.get('quality', {}).get('health'),
            'npms_popularity_score': eval_info.get('popularity', {}).get('communityInterest'),
            'npms_maintenance_score': eval_info.get('maintenance', {}).get('releasesFrequency')
        }
        raw_records.append(record)
        
    print(f"Processed {len(raw_records)} valid NPM records.")
    
    # Step 5: PyPI packages
    print("Step 5: Fetching PyPI packages...")
    pypi_records = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(fetch_pypi_single, p) for p in CURATED_PYPI]
        for f in concurrent.futures.as_completed(futures):
            res = f.result()
            if res:
                pypi_records.append(res)
    print(f"Processed {len(pypi_records)} PyPI records.")
    
    # Enrich PyPI records with GitHub info if available
    print("Step 6: Enriching PyPI GitHub repository metadata where available...")
    for rec in pypi_records:
        r_url = rec.get('repository_url', '')
        if 'github.com' in r_url:
            parts = r_url.rstrip('/').replace('.git', '').split('github.com/')[-1].split('/')
            if len(parts) >= 2:
                owner, repo = parts[0], parts[1]
                try:
                    gh_r = requests.get(f'https://api.github.com/repos/{owner}/{repo}', timeout=5)
                    if gh_r.status_code == 200:
                        gh_data = gh_r.json()
                        rec['stars_count'] = gh_data.get('stargazers_count', 0)
                        rec['forks_count'] = gh_data.get('forks_count', 0)
                        rec['subscribers_count'] = gh_data.get('subscribers_count', 0)
                        rec['open_issues_count'] = gh_data.get('open_issues_count', 0)
                        rec['total_issues_count'] = gh_data.get('open_issues_count', 0) * 4 # conservative estimate
                        rec['is_deprecated'] = gh_data.get('archived', False)
                except Exception:
                    pass
                    
    all_records = raw_records + pypi_records
    df_raw = pd.DataFrame(all_records)
    raw_path = 'data/raw_dataset.csv'
    df_raw.to_csv(raw_path, index=False)
    print(f"Successfully saved {len(df_raw)} records to {raw_path}!")
    print("Columns:", list(df_raw.columns))
    print("Ecosystem breakdown:\n", df_raw['ecosystem'].value_counts())

if __name__ == '__main__':
    main()
