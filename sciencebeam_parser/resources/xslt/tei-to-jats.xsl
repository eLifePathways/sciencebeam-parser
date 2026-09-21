<?xml version="1.0" encoding="UTF-8"?>
<xsl:stylesheet 
  xmlns:xsl="http://www.w3.org/1999/XSL/Transform" 
  xmlns:xs="http://www.w3.org/2001/XMLSchema" 
  xmlns:mml="http://www.w3.org/1998/Math/MathML" 
  xmlns:xlink="http://www.w3.org/1999/xlink" 
  xmlns:tei="http://www.tei-c.org/ns/1.0"
  exclude-result-prefixes="xs mml tei"
  version="1.0"
>
  <xsl:param name="output_parameters" select="'false'"/>
  <xsl:param name="output_bold" select="'false'"/>
  <xsl:param name="output_italic" select="'false'"/>
  <xsl:param name="output_empty_figure_graphic" select="'true'"/>
  <xsl:param name="acknowledgement_target" select="'ack'"/>
  <xsl:param name="annex_target" select="'back'"/>

  <xsl:template match="/">
    <article article-type="research-article">
      <front>
        <xsl:apply-templates select="tei:TEI/tei:teiHeader"/>
      </front>
      <body>
        <xsl:apply-templates select="tei:TEI/tei:text/tei:body"/>
        <xsl:if test="$acknowledgement_target = 'body'">
          <xsl:apply-templates select="tei:TEI/tei:text/tei:back/tei:div[@type='acknowledgement']/tei:div"/>
        </xsl:if>
        <xsl:if test="$annex_target = 'body'">
          <xsl:for-each select="tei:TEI/tei:text/tei:back/tei:div[@type='annex']">
            <xsl:apply-templates select="tei:div"/>
            <xsl:if test="tei:figure">
              <sec id="annex_figures">
                <title>Annex Figures</title>
                <xsl:apply-templates select="tei:figure"/>
              </sec>
            </xsl:if>
          </xsl:for-each>
        </xsl:if>
      </body>
      <back>
        <xsl:apply-templates select="tei:TEI/tei:text/tei:back"/>
      </back>
    </article>
  </xsl:template>

  <xsl:template match="tei:teiHeader">
    <xsl:if test="tei:fileDesc/tei:sourceDesc/tei:biblStruct/tei:monogr/tei:title">
      <journal-meta>
        <journal-title-group>
          <journal-title>
            <xsl:value-of select="tei:fileDesc/tei:sourceDesc/tei:biblStruct/tei:monogr/tei:title"/>
          </journal-title>
        </journal-title-group>
      </journal-meta>
    </xsl:if>

    <article-meta>
      <title-group>
        <article-title>
          <xsl:apply-templates select="tei:fileDesc/tei:titleStmt/tei:title"/>
        </article-title>
      </title-group>

      <xsl:if test="tei:fileDesc/tei:sourceDesc/tei:biblStruct/tei:analytic/tei:author">
        <contrib-group content-type="author">
          <xsl:for-each select="tei:fileDesc/tei:sourceDesc/tei:biblStruct/tei:analytic/tei:author">
            <contrib contrib-type="person">
              <xsl:apply-templates select="tei:persName"/>

              <xsl:if test="tei:email">
                <email>
                  <xsl:value-of select="tei:email"/>
                </email>
              </xsl:if>

              <xsl:if test="tei:affiliation">
                <xref ref-type="aff">
                  <xsl:attribute name='rid'>
                    <xsl:value-of select="tei:affiliation/@key"/>
                  </xsl:attribute>
                </xref>
              </xsl:if>
            </contrib>
          </xsl:for-each>
        </contrib-group>
      </xsl:if>

      <xsl:for-each select="tei:fileDesc/tei:sourceDesc/tei:biblStruct/tei:analytic/tei:author/tei:affiliation">
        <aff>
          <xsl:attribute name='id'>
            <xsl:value-of select="@key"/>
          </xsl:attribute>
          <xsl:if test="tei:orgName[@type='institution']">
            <institution content-type="orgname">
              <xsl:value-of select="tei:orgName[@type='institution']"/>
            </institution>
          </xsl:if>
          <xsl:if test="tei:orgName[@type='department']">
            <institution content-type="orgdiv1">
              <xsl:value-of select="tei:orgName[@type='department']"/>
            </institution>
          </xsl:if>
          <xsl:if test="tei:orgName[@type='laboratory']">
            <institution content-type="orgdiv2">
              <xsl:value-of select="tei:orgName[@type='laboratory']"/>
            </institution>
          </xsl:if>
          <xsl:if test="tei:address/tei:settlement">
            <city>
              <xsl:value-of select="tei:address/tei:settlement"/>
            </city>
          </xsl:if>
          <xsl:if test="tei:address/tei:country">
            <country>
              <xsl:value-of select="tei:address/tei:country"/>
            </country>
          </xsl:if>
        </aff>
      </xsl:for-each>

      <abstract>
        <xsl:apply-templates select="tei:profileDesc/tei:abstract"/>
      </abstract>

      <!-- JATS allows one custom-meta-group, so attribution and the parameters
           share it. The attribution is read from the TEI rather than passed in,
           so the two outputs cannot disagree about what produced them. -->
      <xsl:if test="tei:encodingDesc/tei:appInfo/tei:application or $output_parameters = 'true'">
        <custom-meta-group>
          <xsl:for-each select="tei:encodingDesc/tei:appInfo/tei:application">
            <custom-meta>
              <meta-name>sciencebeam-parser-version</meta-name>
              <meta-value><xsl:value-of select="@version"/></meta-value>
            </custom-meta>
            <xsl:for-each select="tei:label[@type='profile']">
              <custom-meta>
                <meta-name>sciencebeam-parser-profile</meta-name>
                <meta-value><xsl:value-of select="."/></meta-value>
              </custom-meta>
            </xsl:for-each>
            <xsl:for-each select="tei:label[@type='profile-digest']">
              <custom-meta>
                <meta-name>sciencebeam-parser-profile-digest</meta-name>
                <meta-value><xsl:value-of select="."/></meta-value>
              </custom-meta>
            </xsl:for-each>
          </xsl:for-each>
          <xsl:if test="$output_parameters = 'true'">
            <custom-meta>
              <meta-name>xslt-param-acknowledgement_target</meta-name>
              <meta-value><xsl:value-of select="$acknowledgement_target"/></meta-value>
            </custom-meta>
            <custom-meta>
              <meta-name>xslt-param-annex_target</meta-name>
              <meta-value><xsl:value-of select="$annex_target"/></meta-value>
            </custom-meta>
          </xsl:if>
        </custom-meta-group>
      </xsl:if>
    </article-meta>
  </xsl:template>

  <xsl:template match="tei:body">
    <xsl:apply-templates select="tei:div"/>
    <xsl:if test="tei:figure">
      <sec id="figures">
        <title>Figures</title>
        <xsl:apply-templates select="tei:figure"/>
      </sec>
    </xsl:if>
  </xsl:template>

  <xsl:template match="tei:graphic">
    <graphic>
      <xsl:if test="@url">
        <xsl:attribute name='xlink:href'>
          <xsl:value-of select="@url"/>
        </xsl:attribute>
      </xsl:if>
    </graphic>
  </xsl:template>

  <xsl:template match="tei:figure[not(@type='table')]">
    <fig>
      <xsl:attribute name='id'>
        <xsl:value-of select="@xml:id"/>
      </xsl:attribute>
      <object-id><xsl:value-of select="@xml:id"/></object-id>
      <label><xsl:value-of select="tei:head"/></label>
      <caption>
        <xsl:apply-templates select="tei:head"/>
        <p><xsl:apply-templates select="tei:figDesc"/></p>
      </caption>
      <xsl:apply-templates select="tei:graphic"/>
      <xsl:if test="$output_empty_figure_graphic = 'true'">
        <xsl:if test="not(tei:graphic)">
          <graphic/>
        </xsl:if>
      </xsl:if>
    </fig>
  </xsl:template>

  <xsl:template match="tei:figure[@type='table']">
    <table-wrap>
      <xsl:attribute name='id'>
        <xsl:value-of select="@xml:id"/>
      </xsl:attribute>
      <label><xsl:value-of select="tei:head"/></label>
      <caption>
        <xsl:apply-templates select="tei:head"/>
        <p><xsl:apply-templates select="tei:figDesc"/></p>
      </caption>
      <table>
        <tbody>
          <tr>
            <td>
              <xsl:value-of select="tei:table"/>
            </td>
          </tr>
        </tbody>
      </table>
    </table-wrap>
  </xsl:template>

  <xsl:template match="tei:div">
    <sec>
      <xsl:apply-templates select="tei:head"/>
      <xsl:apply-templates select="tei:p"/>
    </sec>
  </xsl:template>

  <xsl:template match="tei:back">
    <xsl:if test="$acknowledgement_target = 'ack'">
      <xsl:if test="tei:div[@type='acknowledgement']">
        <ack>
          <xsl:apply-templates select="tei:div[@type='acknowledgement']/tei:div"/>
        </ack>
      </xsl:if>
    </xsl:if>
    <xsl:if test="$annex_target = 'back'">
      <xsl:for-each select="tei:div[@type='annex']">
        <xsl:apply-templates select="tei:div"/>
        <xsl:if test="tei:figure">
          <sec id="annex_figures">
            <title>Annex Figures</title>
            <xsl:apply-templates select="tei:figure"/>
          </sec>
        </xsl:if>
      </xsl:for-each>
    </xsl:if>
    <xsl:apply-templates select="tei:div/tei:listBibl"/>
    <xsl:if test="$annex_target = 'app'">
      <xsl:if test="tei:div[@type='annex']">
        <app-group>
          <app id="appendix-1">
            <title>Appendix 1</title>
            <xsl:apply-templates select="tei:div[@type='annex']/tei:div"/>
            <xsl:apply-templates select="tei:div[@type='annex']/tei:figure"/>
          </app>
        </app-group>
      </xsl:if>
    </xsl:if>
  </xsl:template>

  <xsl:template match="tei:listBibl">
    <xsl:if test="tei:biblStruct">
      <ref-list id="ref-list-1">
        <xsl:apply-templates select="tei:biblStruct"/>
      </ref-list>
    </xsl:if>
  </xsl:template>

  <xsl:template match="tei:biblStruct">
    <ref>
      <xsl:attribute name='id'>
        <xsl:value-of select="./@xml:id"/>
      </xsl:attribute>

      <element-citation publication-type="journal">
        <xsl:choose>
          <xsl:when test="tei:analytic/tei:title[@type='main']">
            <article-title>
              <xsl:value-of select="tei:analytic/tei:title[@type='main']"/>
            </article-title>
          </xsl:when>
          <xsl:when test="tei:monogr/tei:title[@type='main']">
            <article-title>
              <xsl:value-of select="tei:monogr/tei:title[@type='main']"/>
            </article-title>
          </xsl:when>
        </xsl:choose>

        <xsl:if test="tei:monogr/tei:title[@level='j']">
          <source>
            <xsl:value-of select="tei:monogr/tei:title[@level='j']"/>
          </source>
        </xsl:if>

        <xsl:if test="tei:monogr/tei:imprint/tei:date[@type='published']">
          <xsl:call-template name="parseDateComponents">
            <xsl:with-param name="date" select="tei:monogr/tei:imprint/tei:date[@type='published']/@when"/>
          </xsl:call-template>
        </xsl:if>

        <xsl:if test="tei:monogr/tei:imprint/tei:biblScope[@unit='volume']">
          <volume>
            <xsl:value-of select="tei:monogr/tei:imprint/tei:biblScope[@unit='volume']"/>
          </volume>
        </xsl:if>

        <xsl:if test="tei:monogr/tei:imprint/tei:biblScope[@unit='issue']">
          <issue>
            <xsl:value-of select="tei:monogr/tei:imprint/tei:biblScope[@unit='issue']"/>
          </issue>
        </xsl:if>

        <xsl:choose>
          <xsl:when test="tei:monogr/tei:imprint/tei:biblScope[@unit='page'][@from or @to]">
            <xsl:if test="tei:monogr/tei:imprint/tei:biblScope[@unit='page']/@from">
              <fpage>
                <xsl:value-of select="tei:monogr/tei:imprint/tei:biblScope[@unit='page']/@from"/>
              </fpage>
            </xsl:if>

            <xsl:if test="tei:monogr/tei:imprint/tei:biblScope[@unit='page']/@to">
              <lpage>
                <xsl:value-of select="tei:monogr/tei:imprint/tei:biblScope[@unit='page']/@to"/>
              </lpage>
            </xsl:if>
          </xsl:when>
          <xsl:when test="tei:monogr/tei:imprint/tei:biblScope[@unit='page']/text()">
            <fpage>
              <xsl:value-of select="tei:monogr/tei:imprint/tei:biblScope[@unit='page']"/>
            </fpage>
            <lpage>
              <xsl:value-of select="tei:monogr/tei:imprint/tei:biblScope[@unit='page']"/>
            </lpage>
          </xsl:when>
        </xsl:choose>

        <xsl:if test="tei:monogr/tei:idno[@type='doi']">
          <pub-id pub-id-type="doi">
            <xsl:value-of select="tei:monogr/tei:idno[@type='doi']"/>
          </pub-id>
        </xsl:if>

        <xsl:if test="tei:analytic/tei:author/tei:persName">
          <person-group person-group-type="author">
            <xsl:apply-templates select="tei:analytic/tei:author/tei:persName"/>
          </person-group>
        </xsl:if>

        <xsl:if test="tei:monogr/tei:author/tei:persName">
          <person-group person-group-type="author">
            <xsl:apply-templates select="tei:monogr/tei:author/tei:persName"/>
          </person-group>
        </xsl:if>
      </element-citation>
    </ref>
  </xsl:template>

  <xsl:template match="tei:persName">
    <name>
      <surname>
        <xsl:value-of select="tei:surname"/>
      </surname>
      <given-names>
        <xsl:for-each select="tei:forename">
          <xsl:if test="position() > 1">
            <xsl:value-of select="' '"/>
          </xsl:if>
          <xsl:value-of select="string(.)"/>
        </xsl:for-each>
      </given-names>
    </name>
  </xsl:template>

  <xsl:template match="tei:ref">
    <xsl:choose>
      <xsl:when test="@type = 'bibr' and @target">
        <xref ref-type="bibr">
          <xsl:attribute name='rid'>
            <xsl:value-of select="substring-after(@target, '#')"/>
          </xsl:attribute>
          <xsl:value-of select="."/>
        </xref>
      </xsl:when>
      <xsl:when test="@type = 'figure' and @target">
        <xref ref-type="fig">
          <xsl:attribute name='rid'>
            <xsl:value-of select="substring-after(@target, '#')"/>
          </xsl:attribute>
          <xsl:value-of select="."/>
        </xref>
      </xsl:when>
      <xsl:when test="@type = 'table' and @target">
        <xref ref-type="table">
          <xsl:attribute name='rid'>
            <xsl:value-of select="substring-after(@target, '#')"/>
          </xsl:attribute>
          <xsl:value-of select="."/>
        </xref>
      </xsl:when>
      <xsl:otherwise>
        <xsl:value-of select="."/>
      </xsl:otherwise>
    </xsl:choose>
  </xsl:template>

  <xsl:template match="tei:head">
    <title><xsl:apply-templates select="node()"/></title>
  </xsl:template>

  <xsl:template match="tei:title">
    <xsl:apply-templates select="node()"/>
  </xsl:template>

  <xsl:template match="tei:p">
    <p>
      <xsl:apply-templates select="node()"/>
    </p>
  </xsl:template>

  <xsl:template match="tei:hi[@rend='italic']">
    <xsl:choose>
      <xsl:when test="$output_italic = 'true'">
        <i><xsl:apply-templates select="node()"/></i>
      </xsl:when>
      <xsl:otherwise>
        <xsl:apply-templates select="node()"/>
      </xsl:otherwise>
    </xsl:choose>
  </xsl:template>

  <xsl:template match="tei:hi[@rend='bold']">
    <xsl:choose>
      <xsl:when test="$output_bold = 'true'">
        <b><xsl:apply-templates select="node()"/></b>
      </xsl:when>
      <xsl:otherwise>
        <xsl:apply-templates select="node()"/>
      </xsl:otherwise>
    </xsl:choose>
  </xsl:template>

  <!--
  Template: parseDateComponents (XSLT 1.0)

  Params:
    date: ISO year, year-month or year-month-day
  
  Examples:
    date=2001 => <year>2001</year>
    date=2001-02 => <year>2001</year><month>02</month>
    date=2001-02-03 => <year>2001</year><month>02</month><day>03</day>
  -->
  <xsl:template name="parseDateComponents">
    <xsl:param name="date" select="."/>
    <xsl:call-template name="_parseDateComponentYearMonthDay">
      <xsl:with-param name="value" select="$date"/>
    </xsl:call-template>
  </xsl:template>

  <xsl:template name="_parseDateComponentYearMonthDay">
    <xsl:param name="value" select="."/>
    <xsl:choose>
      <xsl:when test="contains($value, '-')">
        <year>
          <xsl:value-of select="substring-before($value, '-')"/>
        </year>
        <xsl:call-template name="_parseDateComponentMonthDay">
          <xsl:with-param name="value" select="substring-after($value, '-')"/>
        </xsl:call-template>
      </xsl:when>
      <xsl:otherwise>
        <year>
          <xsl:value-of select="$value"/>
        </year>
      </xsl:otherwise>
    </xsl:choose>
  </xsl:template>

  <xsl:template name="_parseDateComponentMonthDay">
    <xsl:param name="value" select="."/>
    <xsl:choose>
      <xsl:when test="contains($value, '-')">
        <month>
          <xsl:value-of select="substring-before($value, '-')"/>
        </month>
        <day>
          <xsl:value-of select="substring-after($value, '-')"/>
        </day>
      </xsl:when>
      <xsl:otherwise>
        <month>
          <xsl:value-of select="$value"/>
        </month>
      </xsl:otherwise>
    </xsl:choose>
  </xsl:template>

</xsl:stylesheet>