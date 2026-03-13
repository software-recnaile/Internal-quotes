// src/components/PDFGenerator.jsx
import React from "react";
import {
  Document,
  Page,
  Text,
  View,
  StyleSheet,
  Image,
} from "@react-pdf/renderer";

// Import your logo - adjust the path based on your project structure
import companyLogo from '../assets/logo-2.png';

const styles = StyleSheet.create({
  page: {
    flexDirection: "column",
    backgroundColor: "#FFFFFF",
    padding: 20,
    fontFamily: "Helvetica",
  },
  headerBox: {
    flexDirection: "row",
    borderBottomWidth: 1,
    borderBottomColor: "#000",
    paddingBottom: 10,
    marginBottom: 15,
  },
  logoColumn: {
    width: "20%",
    paddingRight: 10,
  },
  logo: {
    width: 60,
    height: 60,
    objectFit: "contain",
  },
  addressColumn: {
    width: "50%",
    paddingRight: 10,
    fontSize: 9,
    lineHeight: 1.3,
  },
  titleColumn: {
    width: "30%",
    alignItems: "flex-end",
    justifyContent: "center",
  },
  quotationTitle: {
    fontSize: 16,
    fontWeight: "bold",
    textAlign: "right",
  },
  infoBox: {
    flexDirection: "row",
    justifyContent: "space-between",
    marginBottom: 15,
  },
  quotationInfo: {
    width: "100%",
    fontSize: 10,
  },
  companyInfoBox: {
    marginBottom: 15,
    padding: 10,
    backgroundColor: "#f5f5f5",
    borderRadius: 4,
    fontSize: 9,
  },
  companyInfoTitle: {
    fontWeight: "bold",
    marginBottom: 5,
    fontSize: 10,
  },
  companyInfoRow: {
    flexDirection: "row",
    marginBottom: 3,
  },
  companyInfoLabel: {
    width: "25%",
    fontWeight: "bold",
  },
  companyInfoValue: {
    width: "75%",
  },
  tableHeader: {
    flexDirection: "row",
    borderBottomWidth: 1,
    borderBottomColor: "#000",
    paddingVertical: 5,
    fontSize: 10,
    fontWeight: "bold",
  },
  tableRow: {
    flexDirection: "row",
    borderBottomWidth: 0.5,
    borderBottomColor: "#ccc",
    paddingVertical: 5,
    fontSize: 9,
  },
  snoColumn: {
    width: "5%",
    textAlign: "center",
  },
  fileNameColumn: {
    width: "20%",
    paddingLeft: 5,
  },
  ccColumn: {
    width: "8%",
    textAlign: "center",
  },
  processColumn: {
    width: "15%",
    textAlign: "center",
  },
  amountColumn: {
    width: "10%",
    textAlign: "right",
    paddingRight: 5,
  },
  qtyColumn: {
    width: "8%",
    textAlign: "center",
  },
  perPartCostColumn: {
    width: "12%",
    textAlign: "right",
    paddingRight: 5,
  },
  totalCostColumn: {
    width: "12%",
    textAlign: "right",
    paddingRight: 5,
  },
  summaryRow: {
    flexDirection: "row",
    justifyContent: "flex-end",
    paddingVertical: 5,
    fontSize: 10,
  },
  summaryLabel: {
    width: "70%",
    textAlign: "right",
    paddingRight: 10,
    fontWeight: "bold",
  },
  summaryValue: {
    width: "15%",
    textAlign: "right",
    paddingRight: 5,
    fontSize: 10,
    fontWeight: "bold",
  },
  totalInWords: {
    fontSize: 9,
    marginTop: 10,
    paddingTop: 5,
    borderTopWidth: 0.5,
    borderTopColor: "#ccc",
  },
  termsBox: {
    marginTop: 20,
    fontSize: 8,
    lineHeight: 1.3,
    borderWidth: 0.5,
    borderColor: "#ccc",
    padding: 5,
  },
  termsTitle: {
    fontWeight: "bold",
    marginBottom: 3,
  },
  footer: {
    position: "absolute",
    bottom: 20,
    left: 0,
    right: 0,
    textAlign: "center",
    fontSize: 8,
    color: "#666",
  },
  packingInfo: {
    marginBottom: 10,
    padding: 8,
    backgroundColor: "#e8f4fd",
    borderRadius: 4,
    fontSize: 9,
  },
  packingTitle: {
    fontWeight: "bold",
    marginBottom: 3,
    fontSize: 10,
  },
  packingText: {
    fontSize: 9,
  },
  addressSection: {
    marginBottom: 15,
    padding: 8,
    backgroundColor: "#f0f0f0",
    borderRadius: 4,
    fontSize: 9,
    borderWidth: 1,
    borderColor: "#d0d0d0",
  },
  addressTitle: {
    fontWeight: "bold",
    marginBottom: 5,
    fontSize: 10,
    color: "#333",
  },
  addressRow: {
    flexDirection: "row",
    marginBottom: 3,
  },
  addressLabel: {
    width: "20%",
    fontWeight: "bold",
    fontSize: 9,
    color: "#555",
  },
  addressValue: {
    width: "80%",
    fontSize: 9,
    color: "#333",
  },
});

// Helper function to generate quotation number
const generateQuotationNumber = () => {
  const now = new Date();
  const currentYear = now.getFullYear().toString().slice(-2);
  const nextYear = (now.getFullYear() + 1).toString().slice(-2);
  const uniqueNumber = Math.floor(1000 + Math.random() * 9000);
  return `TDE/${currentYear}-${nextYear}/${uniqueNumber}`;
};

// Helper function to format date
const formatDate = (date) => {
  const day = date.getDate().toString().padStart(2, '0');
  const month = (date.getMonth() + 1).toString().padStart(2, '0');
  const year = date.getFullYear();
  return `${day}/${month}/${year}`;
};

// Helper function to get expiry date (30 days from now)
const getExpiryDate = () => {
  const date = new Date();
  date.setDate(date.getDate() + 30);
  return formatDate(date);
};

// Helper function to convert number to words
const numberToWords = (num) => {
  const units = ['', 'One', 'Two', 'Three', 'Four', 'Five', 'Six', 'Seven', 'Eight', 'Nine'];
  const teens = ['Ten', 'Eleven', 'Twelve', 'Thirteen', 'Fourteen', 'Fifteen', 'Sixteen', 'Seventeen', 'Eighteen', 'Nineteen'];
  const tens = ['', 'Ten', 'Twenty', 'Thirty', 'Forty', 'Fifty', 'Sixty', 'Seventy', 'Eighty', 'Ninety'];
 
  if (num === 0) return 'Zero';
 
  let result = '';
 
  if (num >= 10000000) {
    const crores = Math.floor(num / 10000000);
    result += numberToWords(crores) + ' Crore ';
    num %= 10000000;
  }
 
  if (num >= 100000) {
    const lakhs = Math.floor(num / 100000);
    result += numberToWords(lakhs) + ' Lakh ';
    num %= 100000;
  }
 
  if (num >= 1000) {
    const thousands = Math.floor(num / 1000);
    result += numberToWords(thousands) + ' Thousand ';
    num %= 1000;
  }
 
  if (num >= 100) {
    const hundreds = Math.floor(num / 100);
    result += units[hundreds] + ' Hundred ';
    num %= 100;
  }
 
  if (num >= 20) {
    const ten = Math.floor(num / 10);
    result += tens[ten] + ' ';
    num %= 10;
  } else if (num >= 10) {
    result += teens[num - 10] + ' ';
    num = 0;
  }
 
  if (num > 0) {
    result += units[num] + ' ';
  }
 
  return result.trim() + ' Rupees Only';
};

// Get packing option label
const getPackingLabel = (value) => {
  const options = {
    courier: "Courier",
    porter: "Porter",
    outstation: "Outstation"
  };
  return options[value] || "Not Selected";
};

// Helper function to safely format numbers
const formatCurrency = (value) => {
  if (value === undefined || value === null || value === '') return '0.00';
  const num = parseFloat(value);
  return isNaN(num) ? '0.00' : num.toFixed(2);
};

// Main PDF Generator Component
const QuotationDocument = ({ 
  formData, 
  total, 
  fileCosts,
  companyName = "",
  qrnCode = "",
  gstNo = "",
  shippingAddress = "",
  billingAddress = "",
  packingOption = "",
  discount = 0,
  paintingCost = 0,
  tableRows = [] 
}) => {
  const quotationNumber = generateQuotationNumber();
  const currentDate = formatDate(new Date());
  const expiryDate = getExpiryDate();
 
  // Calculate values based on tableRows or fallback to fileCosts
  let subtotal = 0;
  
  if (tableRows && tableRows.length > 0) {
    // Calculate from tableRows (matching PartsPricingTable calculation)
    subtotal = tableRows.reduce((sum, row) => {
      const totalCost = parseFloat(row.totalCost) || 0;
      return sum + totalCost;
    }, 0);
  } else if (fileCosts) {
    subtotal = fileCosts.reduce((sum, file) => sum + (parseFloat(file?.cost) || 0), 0);
  }
  
  // Add painting cost directly to subtotal
  const paintingCostValue = parseFloat(paintingCost) || 0;
  const subtotalWithPainting = subtotal + paintingCostValue;
  
  // Calculate GST (18% of subtotal with painting)
  const gstRate = 18;
  const gstAmount = subtotalWithPainting * (gstRate / 100);
  
  // Calculate Total (subtotal with painting + GST)
  const totalAmount = subtotalWithPainting + gstAmount;
  
  // Calculate Discount Amount (discount percentage of Total)
  const discountValue = parseFloat(discount) || 0;
  const discountAmount = totalAmount * (discountValue / 100);
  
  // Calculate Grand Total (Total - Discount)
  const grandTotal = totalAmount - discountAmount;

  // Log the props to help debug
  console.log("PDF Generator - Calculations:", {
    subtotal,
    paintingCostValue,
    subtotalWithPainting,
    gstAmount,
    totalAmount,
    discountValue,
    discountAmount,
    grandTotal
  });
 
  return (
    <Document>
      <Page size="A4" style={styles.page}>
        {/* Header Box */}
        <View style={styles.headerBox}>
          <View style={styles.logoColumn}>
            <Image style={styles.logo} src={companyLogo} />
          </View>
          <View style={styles.addressColumn}>
            <Text>3DE TECHNOLOGY PROTOTYPE</Text>
            <Text>SOLUTIONS PVT LTD</Text>
            <Text>No 5, SF No 16/2A1</Text>
            <Text>NEAR SIDCO INDUSTRIAL ESTATE PHASE1</Text>
            <Text>ZUZUVADI</Text>
            <Text>Hosur Tamil Nadu 635126</Text>
            <Text>India</Text>
            <Text>GSTIN 33AABCZ2737P1ZW</Text>
          </View>
          <View style={styles.titleColumn}>
            <Text style={styles.quotationTitle}>Online Quotation</Text>
          </View>
        </View>
       
        {/* Company Information Section */}
        {companyName && (
          <View style={styles.companyInfoBox}>
            <Text style={styles.companyInfoTitle}>Customer Details</Text>
            <View style={styles.companyInfoRow}>
              <Text style={styles.companyInfoLabel}>Company:</Text>
              <Text style={styles.companyInfoValue}>{companyName}</Text>
            </View>
            {gstNo && gstNo !== "" && (
              <View style={styles.companyInfoRow}>
                <Text style={styles.companyInfoLabel}>GST No:</Text>
                <Text style={styles.companyInfoValue}>{gstNo}</Text>
              </View>
            )}
            {qrnCode && qrnCode !== "" && (
              <View style={styles.companyInfoRow}>
                <Text style={styles.companyInfoLabel}>QRN Code:</Text>
                <Text style={styles.companyInfoValue}>{qrnCode}</Text>
              </View>
            )}
          </View>
        )}
       
        {/* Packing Information */}
        {packingOption && packingOption !== "" && (
          <View style={styles.packingInfo}>
            <Text style={styles.packingTitle}>Packing & Forwarding</Text>
            <Text style={styles.packingText}>Mode: {getPackingLabel(packingOption)}</Text>
          </View>
        )}

        {/* Shipping and Billing Address Section */}
        {(shippingAddress && shippingAddress !== "") || (billingAddress && billingAddress !== "") ? (
          <View style={styles.addressSection}>
            <Text style={styles.addressTitle}>Shipping & Billing Address</Text>
            {shippingAddress && shippingAddress !== "" ? (
              <View style={styles.addressRow}>
                <Text style={styles.addressLabel}>Shipping:</Text>
                <Text style={styles.addressValue}>{shippingAddress}</Text>
              </View>
            ) : null}
            {billingAddress && billingAddress !== "" ? (
              <View style={styles.addressRow}>
                <Text style={styles.addressLabel}>Billing:</Text>
                <Text style={styles.addressValue}>{billingAddress}</Text>
              </View>
            ) : null}
          </View>
        ) : null}

        {/* Info Box */}
        <View style={styles.infoBox}>
          <View style={styles.quotationInfo}>
            <Text>Quotation No: {quotationNumber}</Text>
            <Text>Estimate Date: {currentDate}</Text>
            <Text>Expiry Date: {expiryDate}</Text>
          </View>
        </View>
       
        {/* Table Header */}
        <View style={styles.tableHeader}>
          <Text style={styles.snoColumn}>SNO</Text>
          <Text style={styles.fileNameColumn}>PART NAME</Text>
          <Text style={styles.ccColumn}>CC</Text>
          <Text style={styles.processColumn}>Process</Text>
          <Text style={styles.amountColumn}>Amount </Text>
          <Text style={styles.qtyColumn}>QTY</Text>
          <Text style={styles.perPartCostColumn}>Per Part </Text>
          <Text style={styles.totalCostColumn}>Total</Text>
        </View>
       
        {/* Table Rows */}
        {tableRows && tableRows.length > 0 ? (
          tableRows.map((row, index) => (
            <View key={index} style={styles.tableRow}>
              <Text style={styles.snoColumn}>{index + 1}</Text>
              <Text style={styles.fileNameColumn}>{row.partName || `Part ${index + 1}`}</Text>
              <Text style={styles.ccColumn}>{row.cc || '-'}</Text>
              <Text style={styles.processColumn}>{row.process || '-'}</Text>
              <Text style={styles.amountColumn}>
                {row.amount ? `${formatCurrency(row.amount)}` : '-'}
              </Text>
              <Text style={styles.qtyColumn}>{row.quantity || '-'}</Text>
              <Text style={styles.perPartCostColumn}>
                {row.perPartCost ? `${formatCurrency(row.perPartCost)}` : '-'}
              </Text>
              <Text style={styles.totalCostColumn}>
                {row.totalCost ? `${formatCurrency(row.totalCost)}` : '-'}
              </Text>
            </View>
          ))
        ) : (
          // Fallback to fileCosts if tableRows not available
          formData?.files && formData.files.map((file, index) => {
            const fileCost = fileCosts ? fileCosts[index]?.cost || 0 : 0;
            const fileNameWithoutExt = file.fileName.replace(/\.[^/.]+$/, "");
            return (
              <View key={index} style={styles.tableRow}>
                <Text style={styles.snoColumn}>{index + 1}</Text>
                <Text style={styles.fileNameColumn}>{fileNameWithoutExt}</Text>
                <Text style={styles.ccColumn}>{file.volume_cc ? file.volume_cc.toFixed(2) : '-'}</Text>
                <Text style={styles.processColumn}>{file.selectedTechnology?.technology || 'N/A'}</Text>
                <Text style={styles.amountColumn}>-</Text>
                <Text style={styles.qtyColumn}>{file.quantity || 1}</Text>
                <Text style={styles.perPartCostColumn}>{formatCurrency(fileCost)}</Text>
                <Text style={styles.totalCostColumn}>{formatCurrency(fileCost)}</Text>
              </View>
            );
          })
        )}
       
        {/* Summary */}
        <View style={styles.summaryRow}>
          <Text style={styles.summaryLabel}>SUB TOTAL:</Text>
          <Text style={styles.summaryValue}>{formatCurrency(subtotalWithPainting)}</Text>
        </View>
        
        <View style={styles.summaryRow}>
          <Text style={styles.summaryLabel}>GST {gstRate}%:</Text>
          <Text style={styles.summaryValue}>{formatCurrency(gstAmount)}</Text>
        </View>
        
        <View style={styles.summaryRow}>
          <Text style={styles.summaryLabel}>TOTAL:</Text>
          <Text style={styles.summaryValue}>{formatCurrency(totalAmount)}</Text>
        </View>
        
        {/* Discount if applied */}
        {discountValue > 0 && (
          <View style={styles.summaryRow}>
            <Text style={styles.summaryLabel}>DISCOUNT ({discountValue}%):</Text>
            <Text style={styles.summaryValue}>-{formatCurrency(discountAmount)}</Text>
          </View>
        )}
        
        <View style={styles.summaryRow}>
          <Text style={styles.summaryLabel}>GRAND TOTAL:</Text>
          <Text style={styles.summaryValue}>{formatCurrency(grandTotal)}</Text>
        </View>
       
        {/* Total in words */}
        <Text style={styles.totalInWords}>
          Total in words: {numberToWords(Math.round(grandTotal))}
        </Text>
       
        {/* Terms and Conditions */}
        <View style={styles.termsBox}>
          <Text style={styles.termsTitle}>Terms & Conditions</Text>
          <Text>
            1. Please place the purchase order on 3DE TECHNOLOGY PROTOTYPE SOLUTIONS Private Limited Hosur.
          </Text>
          <Text>
            2. Payment Terms : 100% advance along with purchase order by demand draft
          </Text>
          <Text>3. Bank Charges : Not applicable</Text>
          <Text>4. Taxes: Included in grand total</Text>
          <Text>5. Freight & Insurance : Customer Account</Text>
          <Text>
            6. Delivery : 2 - 3 Working days from the date of receipt of your order or Final CAD approval date
          </Text>
          <Text>7. Delivery Terms : Ex-Works Hosur/ Bangalore</Text>
          <Text>8. Despatch Method : {packingOption ? getPackingLabel(packingOption) : 'By Road transport / Courier'}</Text>
          <Text>
            9. Warranty : Not applicable since the quoted items are only Proto Parts
          </Text>
          <Text>10. Validity : This quotation is valid for 30 days</Text>
        </View>
       
        {/* Footer */}
        <Text style={styles.footer} fixed>
          Thank you for your business! • Email: salesupport@3dtechproto.com • Phone: +91 8667291474
        </Text>
      </Page>
    </Document>
  );
};

export default QuotationDocument;