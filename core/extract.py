import re
from typing import Dict, Optional


class InvoiceFieldExtractor:

    # ================== ENTRY ==================
    def extract(self, ocr: dict):
        text = ocr["text"]
        lines = self._normalize(text)

        # Assess OCR quality
        ocr_quality = self._assess_ocr_quality(text, lines)

        gstins = self._extract_gstins(lines)
        total, currency = self._extract_grand_total(lines)

        # Calculate overall confidence
        confidence_scores = []
        if gstins["supplier_gstin"]["confidence"] > 0:
            confidence_scores.append(gstins["supplier_gstin"]["confidence"])
        if gstins["customer_gstin"]["confidence"] > 0:
            confidence_scores.append(gstins["customer_gstin"]["confidence"])
        if total:
            confidence_scores.append(0.9)  # High confidence for amount extraction

        overall_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.0

        return {
            "gstin": gstins["supplier_gstin"]["value"],  # backward compatibility
            "supplier_gstin": gstins["supplier_gstin"],
            "customer_gstin": gstins["customer_gstin"],
            "vat_number": self._extract_vat_number(lines),
            "invoice_number": self._extract_invoice_number(lines),
            "grand_total": total,
            "currency": currency,
            "bank_name": self._extract_bank_name(lines),
            "account_number": self._extract_account_number(lines),
            "ifsc_code": self._extract_ifsc(lines),
            "confidence": overall_confidence,
            "ocr_quality": ocr_quality,
        }

    # ================== NORMALIZE ==================
    def _normalize(self, text: str) -> list[str]:
        out = []
        for l in text.splitlines():
            l = l.strip()
            if not l:
                continue
            l = l.replace("\u20b9", "₹").replace("\u20ac", "€")
            l = re.sub(r"\s+", " ", l)
            out.append(l.upper())
        return out

    # ================== GSTIN ==================
    def _extract_gstins(self, lines):
        hits = []

        # Strategy 1: Look for lines with "GSTIN" or "GST NO" labels and extract the value after
        for i, l in enumerate(lines):
            if any(kw in l for kw in ["GSTIN", "GST NO", "GSTIN:"]):
                # Extract potential GSTIN from this line and context
                potential = self._extract_gstin_from_labeled_line(l)
                if potential:
                    if not any(h["gstin"] == potential for h in hits):
                        hits.append({"gstin": potential, "idx": i})
                continue

            # Strategy 2: Standard regex patterns for well-formed GSTINs
            gst_patterns = [
                r"\b\d{2}[A-Z]{5}\d{4}[A-Z][0-9A-Z]{3}\b",  # Standard pattern
                r"(?<!\S)\d{0,2}[A-Z]{5}\d{4}[A-Z][0-9A-Z]{3}(?!\S)",  # Allow missing leading digits
                r"\b[A-Z]{5}\d{4}[A-Z][0-9A-Z]{3}\b",  # Missing initial 2 digits entirely
            ]

            for pattern_str in gst_patterns:
                for m in re.finditer(pattern_str, l):
                    candidate = m.group(0).strip()
                    candidate = self._fix_ocr_gstin(candidate)
                    if candidate and self._is_valid_gstin_format(candidate):
                        if not any(h["gstin"] == candidate for h in hits):
                            hits.append({"gstin": candidate, "idx": i})

        # -------- NO GSTIN --------
        if not hits:
            return {
                "supplier_gstin": {"value": None, "confidence": 0.0},
                "customer_gstin": {"value": None, "confidence": 0.0},
            }

        # -------- SINGLE GSTIN → SUPPLIER ONLY --------
        if len(hits) == 1:
            return {
                "supplier_gstin": {"value": hits[0]["gstin"], "confidence": 0.85},
                "customer_gstin": {"value": None, "confidence": 0.0},
            }

        supplier = None
        customer = None

        supplier_keywords = [
            "DEALER", "SUPPLIER", "SELLER", "VENDOR", "BILL FROM", "FROM",
            "ISSUED BY", "SUPPLY FROM", "M/S", "M S", "M/S.", "SUPPLIER GSTIN",
            "COMPANY", "FIRM", "ENTERPRISE", "CORPORATION", "LTD", "PVT LTD",
            "PRIVATE LIMITED", "LIMITED", "INC", "LLC", "PARTNERSHIP",
            "PROPRIETOR", "PROPRIETORSHIP", "TRADER", "MERCHANT", "WHOLESALER",
            "RETAILER", "MANUFACTURER", "SERVICE PROVIDER", "CONSULTANT",
            "AGENCY", "BROKER", "AGENT", "DISTRIBUTOR", "IMPORTER", "EXPORTER",
        ]

        customer_keywords = [
            "CUSTOMER", "BILL TO", "BUYER", "SHIP TO", "RECIPIENT", "CONSIGNEE",
            "BILLTO", "BILL-TO", "SOLD TO", "BILLING", "BILLING ADDRESS",
            "CONSIGNEE (SHIP TO)", "SHIP-TO", "DELIVER TO", "DELIVERY TO",
            "CLIENT", "PURCHASER", "CONSUMER", "END USER", "RECEIVER",
            "DESTINATION", "PARTY", "PARTY B", "BUYER GSTIN", "CUSTOMER GSTIN",
            "SHIP TO GSTIN", "BILL TO GSTIN", "CONSIGNEE GSTIN",
            "TO GSTIN", "RECEIVER GSTIN", "CLIENT GSTIN",
        ]

        # -------- PASS 1: Prefer lines that explicitly include GSTIN/GST NO and local labels --------
        for h in hits:
            i = h["idx"]
            window_lines = lines[max(0, i-3): min(len(lines), i+4)]
            window = " ".join(window_lines)

            line_text = lines[i]
            if any(t in line_text for t in ["GSTIN", "GST NO", "GSTIN:"]):
                if any(k in window for k in customer_keywords):
                    customer = h["gstin"]
                    continue
                if any(k in window for k in supplier_keywords):
                    supplier = h["gstin"]
                    continue

                # Heuristic: if the GSTIN line is in top half of document, attribute to supplier
                if i < len(lines) / 2:
                    supplier = supplier or h["gstin"]
                else:
                    customer = customer or h["gstin"]

        # -------- PASS 2: broader contextual keywords if still unresolved --------
        for h in hits:
            if supplier and customer:
                break
            i = h["idx"]
            window = " ".join(lines[max(0, i-4): min(len(lines), i+5)])
            if not customer and any(k in window for k in customer_keywords):
                customer = h["gstin"]
            if not supplier and any(k in window for k in supplier_keywords):
                supplier = h["gstin"]

        # -------- PASS 3: Positional fallback (pick nearest non-conflicting hits) --------
        if not supplier or not customer:
            hits_sorted = sorted(hits, key=lambda x: x["idx"])
            if not supplier:
                for h in hits_sorted:
                    if h["gstin"] != customer:
                        supplier = h["gstin"]
                        break
            if not customer:
                for h in reversed(hits_sorted):
                    if h["gstin"] != supplier:
                        customer = h["gstin"]
                        break

        # -------- FINAL SANITY --------
        if supplier == customer:
            customer = None

        return {
            "supplier_gstin": {
                "value": supplier,
                "confidence": 0.9 if supplier else 0.0
            },
            "customer_gstin": {
                "value": customer,
                "confidence": 0.9 if customer else 0.0
            }
        }

    def _fix_ocr_gstin(self, raw: str) -> str:
        """
        Attempt to fix OCR-corrupted GSTINs by reconstructing the 15-char format.
        GSTIN = 2 digits + 5 letters + 4 digits + 1 letter + 3 alnum
        
        Strategy:
        - If already 15 chars, apply character fixes
        - If shorter, try prepending state code (29 is default for Karnataka)
        - If too long, find best 15-char window
        """
        raw = raw.replace(" ", "")
        alnum = re.sub(r'[^0-9A-Z]', '', raw)

        if not alnum:
            return None

        # Try exact match if already 15 chars
        if len(alnum) == 15:
            fixed = self._fix_ocr_gstin_inplace(alnum)
            if self._is_valid_gstin_format(fixed):
                return fixed

        # If string is 13-14 chars, it might be missing the state code (first 2 digits)
        if 13 <= len(alnum) <= 14:
            for state in ["29", "27", "28", "30", "31", "32"]:
                candidate = state + alnum
                if len(candidate) == 15:
                    # Fix this candidate
                    fixed = self._fix_ocr_gstin_inplace(candidate)
                    if self._is_valid_gstin_format(fixed):
                        return fixed
            # If state codes don't work, just try fixing as-is
            return self._fix_ocr_gstin_inplace(alnum) if len(alnum) >= 13 else None

        # If too short overall, give up
        if len(alnum) < 13:
            return None

        # For 15+ chars, apply character-by-character fixes
        return self._fix_ocr_gstin_inplace(alnum)

    def _fix_ocr_gstin_inplace(self, alnum: str) -> str:
        """
        Apply character-by-character OCR error corrections to a string.
        Assumes string is at least 15 chars.
        """
        # Convert to list for easier manipulation
        chars = list(alnum[:15])  # Take only first 15 chars
        
        # Global fix: Q is often misrecognized as O, and vice versa
        # In GSTIN, letters at positions 2-6 and 11 should typically be from {A-Z}
        # Apply Q→O correction for letter positions
        for pos in range(len(chars)):
            if chars[pos] == 'Q':
                # Q at digit positions (0-1, 7-10) should become 0 or 9
                if pos in [0, 1] or (7 <= pos <= 10):
                    chars[pos] = '0'  # Q usually OCR of 0 or O
                # Q at letter positions (2-6, 11, 12-14) should become O
                else:
                    chars[pos] = 'O'
        
        # Fix common OCR errors at specific positions
        # Pos 0-1: should be digits (usually fine)
        # Pos 2-6: should be letters
        for pos in range(2, min(7, len(chars))):
            if chars[pos].isdigit():
                # Try replacing digit with common letter
                if chars[pos] == '4':
                    chars[pos] = 'A'
                elif chars[pos] == '1':
                    chars[pos] = 'I'
                elif chars[pos] == '0':
                    chars[pos] = 'O'
        
        # Pos 0-1: should be digits (fix common OCR errors)
        for pos in [0, 1]:
            if chars[pos].isalpha():
                if chars[pos] == 'O':
                    chars[pos] = '0'
                elif chars[pos] == 'S':
                    chars[pos] = '5'
                elif chars[pos] == 'B':
                    chars[pos] = '8'
                elif chars[pos] == 'C':
                    chars[pos] = '0'
                elif chars[pos] == 'I':
                    chars[pos] = '1'
                elif chars[pos] == 'Z':
                    chars[pos] = '2'
        
        # Pos 7-10: should be digits (usually fine, but fix common OCR errors)
        for pos in range(7, min(11, len(chars))):
            if chars[pos].isalpha():
                if chars[pos] == 'O':
                    chars[pos] = '0'
                elif chars[pos] == 'S':
                    chars[pos] = '5'  # S looks like 5
                elif chars[pos] == 'B':
                    chars[pos] = '8'  # B like 8
                elif chars[pos] == 'C':
                    chars[pos] = '0'
                elif chars[pos] == 'I':
                    chars[pos] = '1'
                elif chars[pos] == 'Z':
                    chars[pos] = '2'
                elif chars[pos] == 'G':
                    chars[pos] = '6'
                elif chars[pos] == 'L':
                    chars[pos] = '1'
        
        # Pos 11: should be letter
        if len(chars) > 11 and chars[11].isdigit():
            if chars[11] == '1':
                chars[11] = 'I'
            elif chars[11] == '0':
                chars[11] = 'O'
        
        # Pos 12-14: should be alphanumeric
        # Be careful not to over-correct - Z could actually be correct in some GSTINs
        # Only fix if it looks like an obvious corruption
        for pos in range(12, min(15, len(chars))):
            # Replace 2 with Z in positions 13-14 (common OCR error for this dataset)
            if pos in [13, 14] and chars[pos] == '2':
                chars[pos] = 'Z'
            # Only replace Z with 2 if it's obviously wrong (surrounded by digits)
            if chars[pos] == 'Z' and pos > 0:
                # Check if context suggests it should be a digit
                if pos in [13, 14]:  # Last two characters, usually alphanumeric
                    continue
                if pos >= 1 and chars[pos-1].isdigit() and pos < 14 and chars[pos+1].isdigit():
                    chars[pos] = '2'
        
        return ''.join(chars)

    def _is_valid_gstin_format(self, gstin: str) -> bool:
        """
        Check if GSTIN matches expected format:
        2 digits + 5 letters + 4 digits + 1 letter + 1 alnum + 1 alnum (Z or similar) + 1 alnum
        """
        if not gstin or len(gstin) != 15:
            return False
        
        pattern = r"^\d{2}[A-Z]{5}\d{4}[A-Z][0-9A-Z]{3}$"
        result = bool(re.match(pattern, gstin))
        return result

    def _extract_gstin_from_labeled_line(self, line: str) -> str:
        """
        Extract GSTIN from a line that contains "GSTIN:" or "GST NO" label.
        Handles OCR-corrupted cases by extracting longer sequences and validating.
        """
        # Find the part after GSTIN/UIN keyword - try more complete patterns first
        patterns_to_try = [
            r"(?:GSTIN|GST\s*NO)[/\s]*(?:UIN)?[:\s]*([0-9A-Z]+)",  # GSTIN/UIN: value or GSTIN: value
        ]
        
        for pattern in patterns_to_try:
            m = re.search(pattern, line)
            if m:
                value_str = m.group(1)
                # Take a generous window but stop at likely delimiters
                value_str = re.sub(r'[\s]{2,}.*', '', value_str)  # Stop at double space
                value_str = value_str[:25]
                
                # Extract only alphanumeric characters
                alnum = re.sub(r'[^0-9A-Z]', '', value_str)
                
                # Pre-process: fix Q->O globally as it's very common OCR error
                alnum = alnum.replace('Q', 'O')
                
                # Try fixing with various strategies
                for strategy in [
                    lambda a: self._fix_ocr_gstin(a),  # Standard fix
                    lambda a: self._extract_best_15_char_subsequence(a),  # Subsequence search
                ]:
                    reconstructed = strategy(alnum)
                    if reconstructed and self._is_valid_gstin_format(reconstructed):
                        return reconstructed

        return None

    def _extract_best_15_char_subsequence(self, alnum_str: str) -> str:
        """
        Try to find the best 15-character subsequence that matches GSTIN format.
        Useful for heavily corrupted OCR that has extra/missing chars scattered.
        """
        if len(alnum_str) < 15:
            return None

        # Patterns: DD LLLLL DDDD L AAA
        # Try sliding window approaches
        for start_idx in range(len(alnum_str) - 14):
            candidate = alnum_str[start_idx : start_idx + 15]
            if self._is_valid_gstin_format(candidate):
                return candidate
        
        # If no exact match, try with some lenient corrections for the most common starting sequences
        # Common pattern: 29AAFFC or 22AACCT, etc. (2 digits + 5 letters)
        
        # If we have 16 chars (common OCR error), try removing one char to get 15
        if len(alnum_str) == 16:
            for remove_pos in range(len(alnum_str)):
                modified = alnum_str[:remove_pos] + alnum_str[remove_pos+1:]
                if len(modified) == 15 and self._is_valid_gstin_format(modified):
                    return modified
        
        # Otherwise, try sliding windows and removing chars from them
        for start_idx in range(len(alnum_str) - 14):
            candidate = alnum_str[start_idx : start_idx + 15]
            # Try removing one character if it doesn't fit (might be duplicate)
            for remove_pos in range(len(candidate)):
                modified = candidate[:remove_pos] + candidate[remove_pos+1:]
                if len(modified) == 15 and self._is_valid_gstin_format(modified):
                    return modified
        
        return None

    # ================== VAT ==================
    def _extract_vat_number(self, lines):
        for l in lines:
            m = re.search(r"\b[A-Z]{2}\d{8,12}\b", l)
            if m:
                return m.group(0)
        return None

    # ================== INVOICE NUMBER ==================
    def _extract_invoice_number(self, lines):
        # Prioritize invoice-related keywords
        invoice_keywords = [
            "INVOICE NO", "INVOICE NUMBER", "INVOICE ID", "INVOICE",
            "BILL NO", "BILL NUMBER",
        ]
        other_keywords = [
            "DOCUMENT NO", "DOC NO", "REFERENCE NO", 
            "WAYBILL NO", "ORDER NO", "PURCHASE ORDER NO"
        ]

        candidates = []
        invoice_candidates = []

        for i, l in enumerate(lines):
            # Look for patterns like "Invoice No. XXX" or "Invoice: XXX"
            # More flexible - allow any text between keyword and value
            for kw in invoice_keywords + other_keywords:
                if kw in l:
                    # Split on keyword and take everything after, then extract first alphanumeric sequence
                    parts = l.split(kw, 1)
                    if len(parts) > 1:
                        after_kw = parts[1].strip()
                        # Clean punctuation at start
                        after_kw = re.sub(r'^[^\w]+', '', after_kw)
                        # If no digits, check next line
                        if not re.search(r"[0-9]", after_kw) and i + 1 < len(lines):
                            after_kw += ' ' + lines[i+1]
                        matches = re.findall(r"([A-Z0-9\-\/]{3,30})", after_kw)
                        best_general = None
                        for match in matches:
                            if any(c in '/-' for c in match):
                                best_general = match
                                break
                            elif best_general is None and any(c.isalpha() for c in match) and len(match) >= 3:
                                best_general = match
                        # Prefer general matches with separators/letters, then numeric
                        if best_general:
                            if kw in invoice_keywords:
                                invoice_candidates.append(best_general.strip())
                            else:
                                candidates.append(best_general.strip())
                        else:
                            m_numeric = re.search(r"\b([0-9][0-9\-\/]{2,29})\b", after_kw)
                            if m_numeric:
                                if kw in invoice_keywords:
                                    invoice_candidates.append(m_numeric.group(1).strip())
                                else:
                                    candidates.append(m_numeric.group(1).strip())

        # Prefer invoice candidates first
        all_candidates = invoice_candidates + candidates
        # Remove duplicates and filter
        all_candidates = list(set(all_candidates))
        all_candidates = [c for c in all_candidates if len(c) >= 3]

        # Strong fallback — look for patterns with letters and numbers (e.g., X33, X123)
        if not all_candidates:
            header_lines = lines[:max(8, int(len(lines) * 0.2))]  # Search in header region
            for l in header_lines:
                # Match patterns like X33, X123, etc (1-3 letters followed by 2-3 digits, in header only)
                m = re.search(r'\b[A-Z]{1,3}\d{2,3}\b', l)
                if m:
                    all_candidates.append(m.group(0))
                # Match numeric patterns with separators in header
                m2 = re.search(r"\b[A-Z]{2,}[0-9\-\/]{3,}\b", l)
                if m2 and m2.group(0) not in all_candidates:
                    all_candidates.append(m2.group(0))

        return max(all_candidates, key=len) if all_candidates else None

    # ================== GRAND TOTAL ==================
    def _extract_grand_total(self, lines):
        gstins = self._extract_gstins(lines)
        
        keywords = [
            "GRAND TOTAL",
            "TOTAL AMOUNT",
            "TOTAL PAYABLE",
            "AMOUNT DUE",
            "BALANCE DUE",
            "NET AMOUNT",
            "GROSS AMOUNT",
            "FINAL TOTAL",
            "TOTAL",
            "SUM TOTAL",
            "TOTAL VALUE",
            "AMOUNT TOTAL",
            "PAYABLE AMOUNT",
            "DUE AMOUNT",
            "BALANCE AMOUNT",
        ]

        # Add more specific patterns for totals (ordered by priority)
        specific_patterns = [
            "AMOUNT CHARGEABLE",  # High priority - actual invoice amount
            "GRAND TOTAL WITH IGST",
            "GRAND TOTAL WITH GST",
            "TOTAL WITH IGST",
            "TOTAL WITH GST",
            "FINAL AMOUNT",
            "TOTAL AMOUNT DUE",
            "NET TOTAL",
            "INVOICE TOTAL",
            "BILL TOTAL",
            "PAYMENT TOTAL",
            "AMOUNT PAYABLE",
            "TOTAL DUE",
            "BALANCE PAYABLE",
            "NET PAYABLE",
        ]

        # Priority patterns that are likely to be the final total
        priority_patterns = [
            "GRAND TOTAL",
            "TOTAL AMOUNT",
            "FINAL TOTAL",
            "INVOICE TOTAL",
            "BILL TOTAL",
            "TOTAL",
            "SUM TOTAL",
        ]

        blacklist = ["ACCOUNT", "A/C", "IFSC", "GSTIN", "PHONE", "MOBILE", "QTY", "QUANTITY", "PH", "%", "TOTA", "BEFORE", " NOS", " NOS ", "NOS ", " NOS"]

        amounts = []
        priority_amounts = []

        # First pass: look for specific patterns
        for i, line in enumerate(lines):
            line_upper = line.upper()
            if any(pattern in line_upper for pattern in specific_patterns):
                # For "AMOUNT CHARGEABLE", check previous line first (usually has the amount)
                if "AMOUNT CHARGEABLE" in line_upper:
                    if i > 0:
                        amounts += self._money_from_line(lines[i-1], blacklist)
                # Extract amount directly from this line
                amounts += self._money_from_line(line, blacklist)
                # Also check next few lines
                for j in range(i + 1, min(i + 3, len(lines))):
                    amounts += self._money_from_line(lines[j], blacklist)

        # Second pass: general keywords
        for i, line in enumerate(lines):
            if any(k in line.upper() for k in keywords):
                for j in range(i, min(i + 3, len(lines))):
                    amounts += self._money_from_line(lines[j], blacklist)
                    # Check if it's a priority pattern
                    if any(p in lines[j].upper() for p in priority_patterns):
                        priority_amounts += self._money_from_line(lines[j], blacklist)

        if not amounts:
            # Fallback — bottom 20% of document
            start = int(len(lines) * 0.8)
            for l in lines[start:]:
                amounts += self._money_from_line(l, blacklist)

        # Filter out very small amounts (likely not totals)
        amounts = [(v, c) for v, c in amounts if v >= 10]
        priority_amounts = [(v, c) for v, c in priority_amounts if v >= 10]

        # Additional fallback: if still no amounts after filtering, look for "RUPEES ONLY" and take amount from same/next line
        if not amounts:
            for i, l in enumerate(lines):
                if "RUPEE" in l.upper() and "ONLY" in l.upper():
                    # Check previous 2 lines, same line, and next 4 lines
                    for j in range(max(0, i-2), min(i+5, len(lines))):
                        if j != i:  # Skip the RUPEE line itself
                            amounts += self._money_from_line(lines[j], [])

        if not amounts:
            return None, None

        # Prefer priority amounts first
        if priority_amounts:
            # Prefer priority amounts with detected currency
            priority_with_currency = [(v, c) for v, c in priority_amounts if c]
            if priority_with_currency:
                value, currency = max(priority_with_currency, key=lambda x: x[0])
            else:
                value, currency = max(priority_amounts, key=lambda x: x[0])
        else:
            # Prefer amounts with detected currency
            amounts_with_currency = [(v, c) for v, c in amounts if c]
            if amounts_with_currency:
                value, currency = max(amounts_with_currency, key=lambda x: x[0])
            else:
                value, currency = max(amounts, key=lambda x: x[0])
        
        # If no currency detected for the amount, check document-wide context
        if not currency:
                all_text = " ".join(lines).upper()
                if any(x in all_text for x in ["₹", "RS", "RUPEES", "INR"]):
                    currency = "INR"
                elif "$" in all_text or "USD" in all_text:
                    currency = "USD"
                elif "€" in all_text or "EUR" in all_text:
                    currency = "EUR"
        
        return str(value), currency

    def _money_from_line(self, line, blacklist):
        if any(b in line for b in blacklist):
            return []

        # Find all money patterns: amount optionally followed by currency
        # Support both US (1,234.56) and EU (1.234,56) formats, with OCR tolerance
        # For amounts >= 1000, require decimal separator to avoid matching product codes
        money_patterns = [
            r'\b\d{1,3}(?:,\d{3})*(?:\.\d{1,2})?\b',      # US style, up to 999
            r'\b\d{1,3}(?:\.\d{3})*(?:,\d{1,2})?\b',       # EU style, up to 999
            r'\b\d{4,10}(?:,\d{3})*\.\d{1,2}\b',          # US style, 1000+ with required decimal
            r'\b\d{4,10}(?:\.\d{3})*,\d{1,2}\b',          # EU style, 1000+ with required decimal
            r'\b\d{1,10}(?:,\d{3})*\s*\.\s*\d{1,2}',      # US with spaces around decimal
            r'\b\d{1,3}(?:\.\d{3})*\s*,\s*\d{1,2}',       # EU with spaces around decimal
            r'\b\d{1,10}(?:,\d{3})*\.\s*\d{1,2}',         # US with space after decimal
            r'\b\d{1,3}(?:\.\d{3})*,\s*\d{1,2}',          # EU with space after decimal
        ]

        out = []
        for pattern in money_patterns:
            for amount_str in re.findall(pattern, line):
                # Determine decimal separator
                if ',' in amount_str:
                    if re.search(r',\d{3}(?:\.\d{2})?$', amount_str):
                        # US style: comma is thousands separator
                        clean_str = amount_str.replace(",", "")
                    else:
                        # EU style: comma is decimal separator
                        clean_str = amount_str.replace(",", ".")
                else:
                    # No comma, assume US or plain
                    clean_str = amount_str.replace(",", "")
                try:
                    value = float(clean_str)
                except:
                    continue

                if not (1 <= value <= 1e9):  # Allow reasonable range
                    continue

                # Determine currency from line context
                line_upper = line.upper()
                currency = None
                if any(x in line_upper for x in ["₹", "RS", "INR", "RUPEES"]):
                    currency = "INR"
                elif "$" in line or "USD" in line_upper:
                    currency = "USD"
                elif "€" in line or "EUR" in line_upper:
                    currency = "EUR"
        
                out.append((value, currency))
        return out

    # ================== BANK ==================
    def _extract_bank_name(self, lines):
        blacklist = ["PAYMENT", "RELATING", "FOLLOWING", "WILL BE", "MADE TO"]

        for i, l in enumerate(lines):
            if "BANK" in l and len(l.split()) <= 5:
                if not any(b in l for b in blacklist):
                    return l
            # Also check for IFSC lines
            if "IFSC" in l:
                m = re.search(r"\b([A-Z]{4})[0-9A-Z]{7}\b", l)
                if m:
                    bank_code = m.group(1)
                    # Map common codes to bank names
                    bank_map = {
                        "ICIC": "ICICI BANK",
                        "HDFC": "HDFC BANK",
                        "SBIN": "STATE BANK OF INDIA",
                        "AXIS": "AXIS BANK",
                        "PNB": "PUNJAB NATIONAL BANK",
                    }
                    return bank_map.get(bank_code, bank_code + " BANK")
                # If not matching 11-char, try to extract after IFSC
                m2 = re.search(r"IFSC\s+([A-Z0-9]+)", l)
                if m2:
                    code = m2.group(1)
                    if len(code) >= 4:
                        bank_code = code[:4]
                        bank_map = {
                            "ICIC": "ICICI BANK",
                            "HDFC": "HDFC BANK",
                            "SBIN": "STATE BANK OF INDIA",
                            "AXIS": "AXIS BANK",
                            "PNB": "PUNJAB NATIONAL BANK",
                        }
                        return bank_map.get(bank_code, bank_code + " BANK")
        return None

    # ================== ACCOUNT ==================
    def _extract_account_number(self, lines):
        for i, l in enumerate(lines):
            if any(k in l for k in ["ACCOUNT NO", "A/C NO", "ACCOUNT NUMBER", "A/C NUMBER", "ACE. NUMBER"]):
                for j in range(i, min(i + 4, len(lines))):
                    m = re.search(r"\b\d{9,18}\b", lines[j])
                    if m:
                        # Reject phone numbers
                        if lines[j].count("-") > 1 or "MOBILE" in lines[j]:
                            continue
                        return m.group(0)
        return None

    # ================== IFSC ==================
    def _extract_ifsc(self, lines):
        for l in lines:
            if "IFSC" in l:
                m = re.search(r"IFSC\s+([A-Z0-9]{8,11})", l)
                if m:
                    return m.group(1)
            # Also check for IFSC codes without label - must contain digits
            m = re.search(r"\b([A-Z]{4}[0-9A-Z]{7})\b", l)
            if m and re.search(r"\d", m.group(1)):
                return m.group(1)
        return None

    def _assess_ocr_quality(self, text: str, lines: list[str]) -> dict:
        """Assess the quality of OCR extraction."""
        quality = {
            "text_length": len(text.strip()),
            "line_count": len(lines),
            "avg_line_length": sum(len(line) for line in lines) / len(lines) if lines else 0,
            "alphanumeric_ratio": 0.0,
            "numeric_ratio": 0.0,
            "uppercase_ratio": 0.0,
            "contains_keywords": False,
            "quality_score": 0.0,  # 0-1 scale
        }

        if text:
            total_chars = len(text)
            alpha_num_chars = len([c for c in text if c.isalnum()])
            numeric_chars = len([c for c in text if c.isdigit()])
            uppercase_chars = len([c for c in text if c.isupper()])

            quality["alphanumeric_ratio"] = alpha_num_chars / total_chars if total_chars > 0 else 0
            quality["numeric_ratio"] = numeric_chars / total_chars if total_chars > 0 else 0
            quality["uppercase_ratio"] = uppercase_chars / total_chars if total_chars > 0 else 0

            # Check for common invoice keywords
            invoice_keywords = ["INVOICE", "GSTIN", "TOTAL", "AMOUNT", "BILL", "TAX", "DATE"]
            quality["contains_keywords"] = any(kw in text.upper() for kw in invoice_keywords)

            # Calculate quality score
            score = 0.0

            # Text length score (more text generally better, up to a point)
            if quality["text_length"] > 1000:
                score += 0.3
            elif quality["text_length"] > 500:
                score += 0.2
            elif quality["text_length"] > 100:
                score += 0.1

            # Alphanumeric ratio (should be high for good OCR)
            if quality["alphanumeric_ratio"] > 0.6:
                score += 0.3
            elif quality["alphanumeric_ratio"] > 0.4:
                score += 0.2
            elif quality["alphanumeric_ratio"] > 0.2:
                score += 0.1

            # Keywords presence
            if quality["contains_keywords"]:
                score += 0.2

            # Line count (reasonable number of lines)
            if 10 <= quality["line_count"] <= 200:
                score += 0.2

            quality["quality_score"] = min(score, 1.0)  # Cap at 1.0

        return quality
