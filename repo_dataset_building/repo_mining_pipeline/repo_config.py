REPOS = [
    "spring-projects/spring-framework",
    "apache/hadoop",
    "apache/logging-log4j2",
    "elastic/elasticsearch",
    "apache/kafka",
    "google/guava",
    "ReactiveX/RxJava",
    "netty/netty",
    "iluwatar/java-design-patterns",
    "square/retrofit",
]

KEYWORDS = [
    "fix", "security", "vulnerability", "vuln", "cve", "patch", "exploit", "rce",
    "xss", "sqli", "sql injection", "deserialization", "auth", "authorization",
    "access control", "dos", "leak", "insecure", "tls", "ssl", "certificate",
    "sandbox", "classloader",
]

BASE_DIR = "repo_cache"

COMMITS_CSV = "security_commits.csv"
HUNKS_CSV = "java_patch_hunks.csv"
