
#!/bin/bash

# Define the URLs for the Emerging Threats rule files
RULE_FILES=(
    "https://rules.emergingthreats.net/open/snort-2.9.0/rules/emerging-ftp.rules"
)

# Directory to store downloaded rules
RULES_DIR="emerging_rules"
mkdir -p $RULES_DIR

# Download the rule files
for url in "${RULE_FILES[@]}"; do
    wget -P $RULES_DIR $url
done

# File to store CVE-tagged rules
CVE_RULES_FILE="cve_tagged_rules.rules"
> $CVE_RULES_FILE

# Extract rules containing CVE references
for file in $RULES_DIR/*.rules; do
    grep "reference:cve" $file >> $CVE_RULES_FILE
done

echo "CVE-tagged rules have been extracted to $CVE_RULES_FILE"
